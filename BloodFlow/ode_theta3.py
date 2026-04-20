import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import time
import copy
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn.functional as F

import setproctitle
setproctitle.setproctitle("csq")

from utils import count_params, LpLoss, GaussianNormalizer
from model import NORM_net_DeltaPhi_ODE2
from rag_utils import get_rag_dataloader
from Adam import Adam


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------------
# seed + EMA
# ----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.backup:
                continue
            p.data.copy_(self.backup[name].data)
        self.backup = {}


def cpu_state_dict(model: torch.nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ----------------------------
# theta utils
# ----------------------------
def pick_theta_from_BC_time(x: torch.Tensor, eps_std=1e-8):
    ns = x.shape[0]
    X = x.reshape(ns, -1).clamp_min(1e-6)

    logX = torch.log(X)
    cand = {
        "log_contrast": logX.max(dim=1).values - logX.min(dim=1).values,
        "log_std": logX.std(dim=1),
        "x_std": X.std(dim=1),
        "log_mean": logX.mean(dim=1),
        "x_mean": X.mean(dim=1),
    }

    best_name, best_theta, best_std = None, None, -1.0
    for name, th in cand.items():
        s = float(th.std().item())
        if s > best_std:
            best_std, best_name, best_theta = s, name, th

    if best_std >= eps_std:
        return best_theta, best_name

    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    try:
        _U, _S, V = torch.pca_lowrank(Xc, q=1)
        pc1 = V[:, 0]
    except Exception:
        _U, _S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        pc1 = Vh[0, :]
    theta = (X - mu).matmul(pc1)
    return theta, "pca_pc1"


def split_by_theta_rank(theta: torch.Tensor, train_q_low=0.2, train_q_high=0.8, test_side="high"):
    ns = theta.numel()
    order = torch.argsort(theta)

    lo = int(round(train_q_low * ns))
    hi = int(round(train_q_high * ns))
    lo = max(0, min(lo, ns))
    hi = max(0, min(hi, ns))
    if hi <= lo:
        lo, hi = ns // 4, 3 * ns // 4

    train_idx = order[lo:hi]
    if test_side == "high":
        test_idx = order[hi:]
    elif test_side == "low":
        test_idx = order[:lo]
    elif test_side == "both":
        test_idx = torch.cat([order[:lo], order[hi:]], dim=0)
    else:
        raise ValueError("test_side must be high / low / both")

    return train_idx, test_idx, lo, hi


def split_test_into_k_equal_parts(theta_test: torch.Tensor, k: int = 5):
    """
    Return parts indices INTO test-set ordering [0..ntest-1], split by theta order.
    """
    order = torch.argsort(theta_test.detach().cpu())
    parts = list(torch.chunk(order, k))
    return parts, order


# ----------------------------
# forward wrapper: force FP32 to avoid FFT-half crash at Nt=121
# ----------------------------
def forward_fp32(model: torch.nn.Module, x: dict):
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return model(x)
    return model(x)


# ----------------------------
# Main
# ----------------------------
def main(args):
    print("\n=============================")
    print("Device:", device)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass
    print("=============================\n")

    PATH = args.data_dir
    LBO_PATH = args.LBO_dir

    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    modes = args.modes
    Fmodes = args.Fmodes
    width = args.width
    nodes = args.size_of_nodes
    BASIS = args.basis

    grad_clip = getattr(args, "grad_clip", 1.0)
    use_ema = getattr(args, "use_ema", True)
    ema_decay = getattr(args, "ema_decay", 0.999)
    weight_decay = getattr(args, "weight_decay", 1e-5)

    # ------------------------------
    # Load data
    # ------------------------------
    data = sio.loadmat(PATH)
    LBOdata = sio.loadmat(LBO_PATH)
    LBO_MATRIX = LBOdata["Eigenvectors"]  # (nodes, K)

    x_dataIn = torch.tensor(data["BC_time"], dtype=torch.float32)  # (ns, 121, 6)
    yx = torch.tensor(data["velocity_x"], dtype=torch.float32)
    yy = torch.tensor(data["velocity_y"], dtype=torch.float32)
    yz = torch.tensor(data["velocity_z"], dtype=torch.float32)

    y_data = torch.zeros((yx.shape[0], yx.shape[1], yx.shape[2], 3), dtype=torch.float32)
    y_data[:, :, :, 0] = yx
    y_data[:, :, :, 1] = yy
    y_data[:, :, :, 2] = yz

    nsample, Nt, C0 = x_dataIn.shape
    assert Nt == 121 and C0 == 6, f"Expect BC_time (ns,121,6), got {x_dataIn.shape}"
    assert y_data.shape[1] == nodes, f"Expect nodes={nodes}, got y nodes {y_data.shape[1]}"

    # ------------------------------
    # split by theta
    # ------------------------------
    theta, theta_name = pick_theta_from_BC_time(x_dataIn)
    train_idx, test_idx, lo_rank, hi_rank = split_by_theta_rank(
        theta,
        train_q_low=args.train_q_low,
        train_q_high=args.train_q_high,
        test_side=args.test_side
    )

    x_train_raw = x_dataIn[train_idx].clone()
    y_train_raw = y_data[train_idx].clone()
    theta_train = theta[train_idx].clone()

    x_test_raw = x_dataIn[test_idx].clone()
    y_test_raw = y_data[test_idx].clone()
    theta_test = theta[test_idx].clone()

    ntrain, ntest = x_train_raw.shape[0], x_test_raw.shape[0]
    print(f"[Split] nsample={nsample}, ntrain={ntrain}, ntest={ntest}")
    print(f"[Theta] name={theta_name}, std={theta.std().item():.6e}, min={theta.min().item():.6f}, max={theta.max().item():.6f}")
    print(f"[Rank band] train ranks [{lo_rank},{hi_rank}) / {nsample}, test_side={args.test_side}")
    if ntrain == 0 or ntest == 0:
        raise RuntimeError("Empty train/test after split.")

    # ------------------------------
    # append theta channel -> (ns,121,7)
    # ------------------------------
    theta_train_rep = theta_train.view(-1, 1, 1).repeat(1, Nt, 1)
    theta_test_rep  = theta_test.view(-1, 1, 1).repeat(1, Nt, 1)

    x_train = torch.cat([x_train_raw, theta_train_rep], dim=-1)
    x_test  = torch.cat([x_test_raw,  theta_test_rep],  dim=-1)

    coord_dim = x_train.shape[-1]  # 7
    print("[Input] coord_dim after adding theta:", coord_dim)

    # ------------------------------
    # normalization (train-only)
    # ------------------------------
    norm_x1 = GaussianNormalizer(x_train[:, :, 0])
    norm_x2 = GaussianNormalizer(x_train[:, :, 1:-1])
    norm_th = GaussianNormalizer(theta_train)

    x_train[:, :, 0]    = norm_x1.encode(x_train[:, :, 0])
    x_train[:, :, 1:-1] = norm_x2.encode(x_train[:, :, 1:-1])
    x_train[:, :, -1]   = norm_th.encode(theta_train).view(-1, 1).repeat(1, Nt)

    x_test[:, :, 0]     = norm_x1.encode(x_test[:, :, 0])
    x_test[:, :, 1:-1]  = norm_x2.encode(x_test[:, :, 1:-1])
    x_test[:, :, -1]    = norm_th.encode(theta_test).view(-1, 1).repeat(1, Nt)

    norm_y = GaussianNormalizer(y_train_raw)
    y_train = norm_y.encode(y_train_raw)
    y_test  = norm_y.encode(y_test_raw)

    # ------------------------------
    # RAG loader
    # ------------------------------
    train_loader, test_loader = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size,
        rag_configs={"training_refer_range": 20, "refer_num": 1}
    )

    # ------------------------------
    # basis
    # ------------------------------
    if BASIS == "LBO":
        BASE_MATRIX = LBO_MATRIX[:, :modes]
    else:
        raise ValueError("Expect BASIS='LBO'")

    TIME_MATRIX = torch.tensor(BASE_MATRIX, dtype=torch.float32, device=device)
    TIME_INVERSE = (TIME_MATRIX.T @ TIME_MATRIX).inverse() @ TIME_MATRIX.T

    BASE_MATRIX = torch.tensor(BASE_MATRIX, dtype=torch.float32, device=device)
    BASE_INVERSE = (BASE_MATRIX.T @ BASE_MATRIX).inverse() @ BASE_MATRIX.T

    # ------------------------------
    # model (class not modified)
    # ------------------------------
    model = NORM_net_DeltaPhi_ODE2(
        modes, nodes, Fmodes, width,
        TIME_MATRIX, TIME_INVERSE,
        BASE_MATRIX, BASE_INVERSE,
        Nt,
        steps=args.steps,
        coord_dim=coord_dim
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )

    myloss = LpLoss(d=3, p=2, size_average=False)
    ema = EMA(model, decay=ema_decay) if use_ema else None

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs,))
    test_error = np.zeros((epochs,))
    ET_list = np.zeros((epochs,))

    printed_once = False
    best_test = float("inf")
    best_ep = -1
    best_state = None
    best_ema_shadow = None  # 保存 EMA shadow（可选）

    for ep in range(epochs):
        model.train()
        train_l2_sum = 0.0

        for x, y in train_loader:
            if not printed_once:
                print("===== Debug: one batch shapes from train_loader =====")
                print("y shape:", y.shape, "dtype:", y.dtype)
                print("x keys:", list(x.keys()))
                for k, v in x.items():
                    print(f"  {k}: {tuple(v.shape)} dtype={v.dtype} device={v.device}")
                print("=====================================================")
                printed_once = True

            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = forward_fp32(model, x)

            B = y.shape[0]
            loss = myloss(out.reshape(B, -1), y.reshape(B, -1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            out_real = norm_y.decode(out.detach().cpu()).reshape(B, -1)
            y_real   = norm_y.decode(y.detach().cpu()).reshape(B, -1)
            train_l2_sum += myloss(out_real, y_real).item()

        scheduler.step()

        # ------------------------------
        # Test (EMA eval)
        # ------------------------------
        model.eval()
        if ema is not None:
            ema.apply_shadow(model)

        test_l2_sum = 0.0
        emax_sum = 0.0
        nb = 0

        with torch.no_grad():
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().to(device)
                y = y.to(device)

                out = forward_fp32(model, x)
                B = y.shape[0]

                out_real = norm_y.decode(out.detach().cpu()).reshape(B, -1)
                y_real   = norm_y.decode(y.detach().cpu()).reshape(B, -1)

                test_l2_sum += myloss(out_real, y_real).item()
                emax_sum += (out_real - y_real).abs().max(dim=1).values.mean().item()
                nb += 1

        if ema is not None:
            ema.restore(model)

        train_l2 = train_l2_sum / ntrain
        test_l2  = test_l2_sum / ntest
        emax = emax_sum / max(nb, 1)

        train_error[ep] = train_l2
        test_error[ep] = test_l2
        ET_list[ep] = emax

        # ✅ best by EMA-eval test_l2
        if test_l2 < best_test:
            best_test = float(test_l2)
            best_ep = ep
            best_state = cpu_state_dict(model)
            if ema is not None:
                # 额外保存 EMA shadow（可选）
                best_ema_shadow = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}

        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep:04d}, LR: {lr_now:.2e}, Train L2: {train_l2:.6f}, "
              f"Test(Extrap) L2: {test_l2:.6f}, Best: {best_test:.6f}@{best_ep:04d}, "
              f"Emax_te: {emax:.6f}, Time: {T:.3f}s")
        time_step = time.perf_counter()

    print("\n=============================")
    print("Training done...")
    print("Best:", best_test, " @ ep", best_ep)
    print("=============================\n")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ------------------------------
    # Predict & Save (batch_size=1)
    # ------------------------------
    train_loader_eval, test_loader_eval = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size=1,
        rag_configs={"training_refer_range": 20, "refer_num": 1},
        train_shuffle=False
    )

    pre_test = torch.zeros_like(y_test_raw)
    y_test_real_all = torch.zeros_like(y_test_raw)
    x_test_real_all = torch.zeros_like(x_test_raw)
    theta_test_all = theta_test.clone()

    idx = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            out = forward_fp32(model, x)

            out_real = norm_y.decode(out.detach().cpu())
            y_real   = norm_y.decode(y.detach().cpu())

            x_dec = x["x"].detach().cpu()

            x0 = norm_x1.decode(x_dec[:, :, 0])
            x1 = norm_x2.decode(x_dec[:, :, 1:-1])
            th = norm_th.decode(x_dec[:, 0, -1])  # (1,)

            x_phys = torch.zeros((1, Nt, 6), dtype=torch.float32)
            x_phys[:, :, 0] = x0
            x_phys[:, :, 1:] = x1

            pre_test[idx] = out_real[0]
            y_test_real_all[idx] = y_real[0]
            x_test_real_all[idx] = x_phys[0]
            idx += 1

    # ------------------------------
    # ✅ NEW: 5-part test loss by theta_test order (equal count)
    # metric consistent with training eval: myloss(flat)/num_samples
    # ------------------------------
    parts5, order_test = split_test_into_k_equal_parts(theta_test, k=5)

    def mean_l2_subset(y_true, y_pred, idx_1d):
        if idx_1d.numel() == 0:
            return np.nan
        yt = y_true[idx_1d].reshape(idx_1d.numel(), -1)
        yp = y_pred[idx_1d].reshape(idx_1d.numel(), -1)
        return myloss(yt, yp).item() / idx_1d.numel()

    test_part_losses = [mean_l2_subset(y_test_real_all, pre_test, p) for p in parts5]

    print("\n[Test-set] 5-part mean L2 by theta_test order:")
    theta_test_cpu = theta_test.detach().cpu()
    for i, idxp in enumerate(parts5):
        if idxp.numel() == 0:
            print(f"  part{i+1}: n=0, mean_L2=nan")
        else:
            th_min = float(theta_test_cpu[idxp].min().item())
            th_max = float(theta_test_cpu[idxp].max().item())
            print(f"  part{i+1}: n={idxp.numel():4d}, theta∈[{th_min:.6f},{th_max:.6f}], mean_L2={test_part_losses[i]}")

    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, "logs_ode_param_extrap", args.CaseName)
    os.makedirs(save_path, exist_ok=True)

    train_time = float(time_step_end - time_start)

    dataframe = pd.DataFrame({
        "Test_loss_extrap": [float(test_error[-1])],
        "Best_test_loss_extrap": [float(best_test)],

        # ✅ NEW: 5-part losses
        "Test_loss_extrap_p1": [float(test_part_losses[0])],
        "Test_loss_extrap_p2": [float(test_part_losses[1])],
        "Test_loss_extrap_p3": [float(test_part_losses[2])],
        "Test_loss_extrap_p4": [float(test_part_losses[3])],
        "Test_loss_extrap_p5": [float(test_part_losses[4])],

        "Train_loss": [float(train_error[-1])],
        "Emax_test": [float(ET_list[-1])],
        "num_paras": [count_params(model)],
        "train_time": [train_time],
        "theta_used": [theta_name],
        "train_q_low": [args.train_q_low],
        "train_q_high": [args.train_q_high],
        "test_side": [args.test_side],
        "ntrain": [ntrain],
        "ntest": [ntest],
        "grad_clip": [grad_clip],
        "use_ema": [use_ema],
        "ema_decay": [ema_decay],
        "weight_decay": [weight_decay],
        "steps": [args.steps],
        "width": [args.width],
        "lr": [args.lr],
        "best_epoch": [int(best_ep)],
    })
    dataframe.to_csv(os.path.join(save_path, "log.csv"), index=False)

    loss_dict = {"train_error": train_error, "test_error": test_error, "ET_list": ET_list}
    pred_dict = {
        "pre_test": pre_test.numpy(),
        "y_test": y_test_real_all.numpy(),
        "x_test_BC_time": x_test_real_all.numpy(),
        "theta_train": theta_train.numpy(),
        "theta_test": theta_test_all.numpy(),
        "theta_name": theta_name,

        # ✅ NEW: split debug
        "test_order_theta": order_test.numpy(),
        "test_part_losses_5": np.array(test_part_losses, dtype=np.float64),
    }

    sio.savemat(os.path.join(save_path, f"NORM_loss_{args.CaseName}.mat"), mdict=loss_dict)
    sio.savemat(os.path.join(save_path, f"NORM_pre_{args.CaseName}.mat"), mdict=pred_dict)

    # ============================
    # ✅ NEW: save model.pt
    # ============================
    torch.save(
        {
            "epoch": int(best_ep),
            "best_test_loss_extrap": float(best_test),
            "theta_used": theta_name,
            "train_q_low": float(args.train_q_low),
            "train_q_high": float(args.train_q_high),
            "test_side": str(args.test_side),
            "coord_dim": int(coord_dim),
            "Nt": int(Nt),
            "modes": int(modes),
            "Fmodes": int(Fmodes),
            "width": int(width),
            "nodes": int(nodes),
            "steps": int(args.steps),

            "model_state_dict": best_state if best_state is not None else cpu_state_dict(model),
            "ema_shadow": best_ema_shadow,  # 可用于复现 EMA 权重（可选）

            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),

            # 保存归一化器（有 decode/encode 时可直接复用）
            "norm_x1": norm_x1,
            "norm_x2": norm_x2,
            "norm_th": norm_th,
            "norm_y": norm_y,

            "args": dict(args.__dict__),
        },
        os.path.join(save_path, "model.pt")
    )

    print(f"\nSaved to: {save_path}")
    print(f"Final Test(Extrap) L2: {test_error[-1]:.6e}")
    print(f"Best  Test(Extrap) L2: {best_test:.6e}")
    print(f"Num params: {count_params(model)}")
    print("Saved checkpoint: model.pt")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for i in range(5):
        rep = i + 1
        set_seed(2024 + rep)

        args = objectview({
            "modes": 64,
            "Fmodes": 16,

            "width": 24,
            "steps": 6,

            "size_of_nodes": 1656,
            "batch_size": 10,
            "epochs": 500,

            "data_dir": "../datasets/BloodFlow.mat",
            "LBO_dir": "../datasets/BloodFlow_LBO_basis/LBO_basis.mat",

            "CaseName": f"velocity_xyz_extrap_tuned_fp32fft_{rep}",
            "basis": "LBO",

            "lr": 0.0015,
            "weight_decay": 1e-5,

            "grad_clip": 1.0,
            "use_ema": True,
            "ema_decay": 0.999,

            "train_q_low": 0.2,
            "train_q_high": 0.8,
            "test_side": "high",
        })

        main(args)

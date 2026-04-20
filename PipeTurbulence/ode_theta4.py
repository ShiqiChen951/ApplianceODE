"""
Parameter Extrapolation Training Script (Index-based split, robust theta, device-safe)
✅ Modified:
- train_q_low=0.0, train_q_high=0.5
- test set split into 5 equal-count parts by theta_test rank
- write 5 part losses into log.csv (Test_loss_p1~p5)
✅ NEW:
- save model.pt (best), model_best.pt, model_last.pt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F

import setproctitle
setproctitle.setproctitle("csq")

from lapy import TriaMesh, Solver

from utils import count_params, LpLoss, GaussianNormalizer
from rag_utils import get_rag_dataloader
from model import Approximation_block


# ----------------------------
# Device
# ----------------------------
def get_device(gpu_id=None):
    if torch.cuda.is_available():
        try:
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
            _ = torch.empty(1, device="cuda")  # quick check
            return torch.device("cuda")
        except Exception as e:
            print("[WARN] CUDA not usable, fallback to CPU. Reason:", str(e))
            return torch.device("cpu")
    return torch.device("cpu")


def cpu_state_dict(model: nn.Module):
    """Return a CPU-cloned state_dict for portable saving."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ----------------------------
# Theta utilities (robust)
# ----------------------------
def build_theta_candidates(c_field: torch.Tensor):
    a = c_field.clamp_min(1e-6)
    loga = torch.log(a)

    thetas = {}
    thetas["log_contrast"] = loga.max(dim=1).values - loga.min(dim=1).values
    thetas["loga_std"]     = loga.std(dim=1)
    thetas["a_std"]        = a.std(dim=1)
    thetas["loga_mean"]    = loga.mean(dim=1)
    thetas["a_mean"]       = a.mean(dim=1)
    return thetas


def pca_theta(c_field: torch.Tensor, max_samples_for_pca=1200):
    X = c_field
    ns, _N = X.shape

    if ns > max_samples_for_pca:
        idx = torch.randperm(ns)[:max_samples_for_pca]
        X_fit = X[idx]
    else:
        X_fit = X

    mu = X_fit.mean(dim=0, keepdim=True)
    Xc_fit = X_fit - mu

    try:
        _U, _S, V = torch.pca_lowrank(Xc_fit, q=1)
        pc1 = V[:, 0]
    except Exception:
        _U, _S, Vh = torch.linalg.svd(Xc_fit, full_matrices=False)
        pc1 = Vh[0, :]

    theta = (X - mu).matmul(pc1)
    return theta


def pick_non_degenerate_theta(c_field: torch.Tensor, eps_std=1e-8):
    cand = build_theta_candidates(c_field)

    best_name, best_theta, best_std = None, None, -1.0
    for name, th in cand.items():
        th_std = float(th.std().item())
        if th_std > best_std:
            best_std = th_std
            best_name = name
            best_theta = th

    if best_std < eps_std:
        return pca_theta(c_field), "pca_pc1"

    return best_theta, best_name


def make_extrapolation_split_by_rank(
    theta: torch.Tensor,
    train_frac_low=0.2, train_frac_high=0.8,
    test_side="high"
):
    ns = theta.numel()
    order = torch.argsort(theta)  # ascending
    lo = int(round(train_frac_low * ns))
    hi = int(round(train_frac_high * ns))

    lo = max(0, min(lo, ns))
    hi = max(0, min(hi, ns))
    if hi <= lo:
        lo = ns // 4
        hi = 3 * ns // 4
        if hi <= lo:
            lo = 0
            hi = ns

    train_idx = order[lo:hi]

    if test_side == "high":
        test_idx = order[hi:]
    elif test_side == "low":
        test_idx = order[:lo]
    elif test_side == "both":
        test_idx = torch.cat([order[:lo], order[hi:]], dim=0)
    else:
        raise ValueError("test_side must be one of: high, low, both")

    theta_sorted = theta[order]
    return train_idx, test_idx, lo, hi, theta_sorted


def split_into_k_equal_parts_by_theta(theta_vec: torch.Tensor, k: int = 5):
    """
    theta_vec: (n,)
    Return:
      parts: list of index tensors (into 0..n-1), each part equal-count (torch.chunk)
      order: theta-sorted indices
    """
    order = torch.argsort(theta_vec.detach().cpu())
    parts = list(torch.chunk(order, k))
    return parts, order


# ----------------------------
# Model (as you pasted)
# ----------------------------
class NORM_Net_ODE2(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE, steps=10, coord_dim=1):
        super(NORM_Net_ODE2, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(3, self.width)
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc3 = nn.Linear(4, self.width)
        self.coord_proj = nn.Linear(2, self.width)

        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.steps = steps
        self.coord_dim = coord_dim

    def func(self, a, h):
        ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
        ha = self.ha_conv(ha)           # (B, width, N)
        h = F.gelu(ha)

        x1 = self.conv0(h)
        x2 = self.w0(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv1(h)
        x2 = self.w1(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv2(h)
        x2 = self.w2(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv3(h)
        x2 = self.w3(h)
        dhdt = x1 + x2

        dhdt = self.dhdt_expand(dhdt)  # (B, 2*width, N)
        dhdt_h, _dhdt_a = torch.split(dhdt, self.width, dim=1)
        return dhdt_h

    def forward(self, x):
        x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        B, N = x_coord.shape[0], x_coord.shape[1]

        ref_score = x['ref_score'].view(B, 1, 1).repeat(1, N, 1)

        ref_y = x['ref_y']
        if ref_y.dim() == 2:
            ref_y = ref_y.unsqueeze(-1)

        grid = self.get_grid((B, N, 1), x['x'].device)

        x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 3)
        x_feat = self.fc0(x_in)                              # (B, N, width)

        a_input = torch.cat([ref_x, x_coord], dim=-1)        # (B, N, ?)
        a_feat = self.fc3(a_input)                           # (B, N, width)

        depth_coord = (x_coord - ref_x) / float(self.steps)  # (B, N, coord_dim)
        depth_feat = self.coord_proj(depth_coord)            # (B, N, width)
        depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)

        h = x_feat.permute(0, 2, 1).contiguous()    # (B, width, N)
        a = a_feat.permute(0, 2, 1).contiguous()    # (B, width, N)

        for _ in range(self.steps):
            k1 = self.func(a, h)
            k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
            k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
            k4 = self.func(a + depth_scale, h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x_out = h.permute(0, 2, 1).contiguous()     # (B, N, width)
        x_out = F.gelu(self.fc1(x_out))             # (B, N, 128)
        x_out = self.fc2(x_out)                     # (B, N, 1)

        return x_out + ref_y.reshape(x_out.shape)   # (B, N, 1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# ----------------------------
# Main
# ----------------------------
def main(args):
    device = get_device(getattr(args, "gpu_id", None))

    print("\n=============================")
    print("Device:", device)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        except Exception:
            pass
    print("=============================\n")

    PATH = args.data_dir
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    modes = args.modes
    width = args.width
    s = args.size_of_nodes

    # ------------------------------
    # Load data + LBO basis
    # ------------------------------
    data = sio.loadmat(PATH)

    k = 128
    Points = np.vstack((data["nodes"].T, np.zeros(s).reshape(1, -1)))
    mesh = TriaMesh(Points.T, data["elements"].T - 1)
    fem = Solver(mesh)
    _evals, LBO_MATRIX = fem.eigs(k=k)

    x_dataIn = torch.tensor(data["Input"], dtype=torch.float32)   # (nsample, N)
    y_dataIn = torch.tensor(data["Output"], dtype=torch.float32)  # (nsample, N)

    nsample, N = x_dataIn.shape
    assert N == s, f"Expected N={s}, got N={N}"

    # ------------------------------
    # theta + rank split (extrapolation)
    # ------------------------------
    theta, theta_name = pick_non_degenerate_theta(x_dataIn)
    train_idx, test_idx, lo_rank, hi_rank, theta_sorted = make_extrapolation_split_by_rank(
        theta,
        train_frac_low=args.train_q_low,
        train_frac_high=args.train_q_high,
        test_side=args.test_side
    )

    x_train_raw = x_dataIn[train_idx]
    y_train_raw = y_dataIn[train_idx]
    theta_train = theta[train_idx]

    x_test_raw = x_dataIn[test_idx]
    y_test_raw = y_dataIn[test_idx]
    theta_test = theta[test_idx]

    ntrain, ntest = x_train_raw.shape[0], x_test_raw.shape[0]
    print(f"[Split] nsample={nsample}, ntrain={ntrain}, ntest={ntest}")
    print(f"[Theta] theta_name={theta_name}, std={theta.std().item():.6e}, min={theta.min().item():.6f}, max={theta.max().item():.6f}")
    print(f"[Rank band] train ranks [{lo_rank}, {hi_rank}) / {nsample}, test_side={args.test_side}")
    if ntrain == 0 or ntest == 0:
        raise RuntimeError("Split produced empty train or test set. Adjust train_q_low/high or test_side.")

    # ------------------------------
    # Build x = [Input, theta] as 2 channels: (nsample, N, 2)
    # separate normalization (Input + theta)
    # ------------------------------
    norm_in = GaussianNormalizer(x_train_raw)  # fit only on train
    x_train_in = norm_in.encode(x_train_raw)
    x_test_in  = norm_in.encode(x_test_raw)

    theta_mu = theta_train.mean()
    theta_std = theta_train.std().clamp_min(1e-8)
    theta_train_z = (theta_train - theta_mu) / theta_std
    theta_test_z  = (theta_test  - theta_mu) / theta_std

    theta_train_node = theta_train_z.view(-1, 1).repeat(1, N)
    theta_test_node  = theta_test_z.view(-1, 1).repeat(1, N)

    x_train = torch.stack([x_train_in, theta_train_node], dim=-1)  # (ntrain, N, 2)
    x_test  = torch.stack([x_test_in,  theta_test_node],  dim=-1)  # (ntest,  N, 2)

    norm_y = GaussianNormalizer(y_train_raw)
    y_train = norm_y.encode(y_train_raw)
    y_test  = norm_y.encode(y_test_raw)

    # ------------------------------
    # RAG dataloaders
    # ------------------------------
    train_loader, test_loader = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size,
        rag_configs={"training_refer_range": 20, "refer_num": 1}
    )

    # ------------------------------
    # Basis + model (device-safe)
    # ------------------------------
    BASE_MATRIX = torch.tensor(LBO_MATRIX[:, :modes], dtype=torch.float32, device=device)
    BASE_INVERSE = (BASE_MATRIX.T @ BASE_MATRIX).inverse() @ BASE_MATRIX.T

    model = NORM_Net_ODE2(modes, width, BASE_MATRIX, BASE_INVERSE, coord_dim=1).to(device)

    # ------------------------------
    # Optimizer + Scheduler
    # ------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )

    myloss = LpLoss(size_average=False)

    # ------------------------------
    # Train (with best checkpoint)
    # ------------------------------
    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs,))
    test_error = np.zeros((epochs,))
    ET_list = np.zeros((epochs,))

    best_test = float("inf")
    best_state = None
    best_ep = -1

    USE_RELATIVE_REAL_LOSS = True

    # ✅ NEW: keep last_state if you want strict last checkpoint
    last_state = None

    for ep in range(epochs):
        model.train()
        train_l2_sum = 0.0

        for x, y in train_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            B = y.shape[0]

            out_real = norm_y.decode(out.view(B, -1))
            y_real   = norm_y.decode(y.view(B, -1))

            if USE_RELATIVE_REAL_LOSS:
                denom = (y_real.norm(p=2, dim=1).mean() + 1e-12)
                loss = myloss(out_real, y_real) / denom
            else:
                loss = myloss(out_real, y_real)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_l2_sum += myloss(out_real.detach(), y_real.detach()).item()

        scheduler.step()

        # ------------------------------
        # Test (REAL space)
        # ------------------------------
        model.eval()
        test_l2_sum = 0.0
        loss_max_test_sum = 0.0
        num_test_batches = 0

        with torch.no_grad():
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().to(device)
                y = y.to(device)

                out = model(x)
                B = y.shape[0]

                out_real = norm_y.decode(out.view(B, -1))
                y_real   = norm_y.decode(y.view(B, -1))
                test_l2_sum += myloss(out_real, y_real).item()

                loss_max_batch = (out.view(B, -1) - y.view(B, -1)).abs().max(dim=1).values.mean().item()
                loss_max_test_sum += loss_max_batch
                num_test_batches += 1

        train_l2 = train_l2_sum / ntrain
        test_l2  = test_l2_sum / ntest
        loss_max_test = loss_max_test_sum / max(num_test_batches, 1)

        train_error[ep] = train_l2
        test_error[ep] = test_l2
        ET_list[ep] = loss_max_test

        # ✅ record best
        if test_l2 < best_test:
            best_test = test_l2
            best_ep = ep
            best_state = cpu_state_dict(model)  # CPU copy

        # ✅ record last (strict last)
        last_state = cpu_state_dict(model)

        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        print(
            f"Step: {ep:04d}, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}, "
            f"Best: {best_test:.6f}@{best_ep:04d}, Emax_test: {loss_max_test:.6f}, Time: {T:.3f}s"
        )
        time_step = time.perf_counter()

    print("\n=============================")
    print("Training done...")
    print(f"Best checkpoint: epoch={best_ep}, best_test={best_test:.6f}")
    print("=============================\n")

    # Load best for evaluation
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

    # ------------------------------
    # Eval predictions with batch_size=1
    # ------------------------------
    train_loader_eval, test_loader_eval = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size=1,
        rag_configs={"training_refer_range": 20, "refer_num": 1},
        train_shuffle=False
    )

    pre_train = torch.zeros((ntrain, N))
    y_train_real_all = torch.zeros((ntrain, N))

    idx = 0
    with torch.no_grad():
        for x, y in train_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())

            pre_train[idx, :] = out_real
            y_train_real_all[idx, :] = y_real
            idx += 1

    pre_test = torch.zeros((ntest, N))
    y_test_real_all = torch.zeros((ntest, N))

    idx = 0
    with torch.no_grad():
        for x, y in test_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())

            pre_test[idx, :] = out_real
            y_test_real_all[idx, :] = y_real
            idx += 1

    # ------------------------------
    # test set split into 5 equal parts by theta_test (rank)
    # Metric: myloss(y[idx], pred[idx]).item() / idx.numel()
    # ------------------------------
    parts5_test, order_test = split_into_k_equal_parts_by_theta(theta_test, k=5)

    def mean_l2_subset(y_true, y_pred, idx_1d):
        if idx_1d.numel() == 0:
            return np.nan
        return myloss(y_true[idx_1d], y_pred[idx_1d]).item() / idx_1d.numel()

    test_part_losses = [mean_l2_subset(y_test_real_all, pre_test, p) for p in parts5_test]

    # ------------------------------
    # Save
    # ------------------------------
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, "logs_ode_real", args.CaseName)
    os.makedirs(save_path, exist_ok=True)

    total_time = time_step_end - time_start

    dataframe = pd.DataFrame({
        "Test_loss_best": [float(best_test)],
        "Best_epoch": [int(best_ep)],
        "Test_loss_last": [float(test_error[-1])],
        "Train_loss_last": [float(train_error[-1])],
        "Emax_test_last": [float(ET_list[-1])],

        "Test_loss_p1": [float(test_part_losses[0])],
        "Test_loss_p2": [float(test_part_losses[1])],
        "Test_loss_p3": [float(test_part_losses[2])],
        "Test_loss_p4": [float(test_part_losses[3])],
        "Test_loss_p5": [float(test_part_losses[4])],

        "num_paras": [count_params(model)],
        "train_time": [float(total_time)],
        "theta_used": [theta_name],
        "train_frac_low": [args.train_q_low],
        "train_frac_high": [args.train_q_high],
        "test_side": [args.test_side],
        "rank_lo": [int(lo_rank)],
        "rank_hi": [int(hi_rank)],
        "theta_train_min": [float(theta_train.min().item())],
        "theta_train_max": [float(theta_train.max().item())],
        "theta_test_min": [float(theta_test.min().item())],
        "theta_test_max": [float(theta_test.max().item())],
        "device": [str(device)],
        "lr": [float(learning_rate)],
        "optimizer": ["AdamW"],
        "scheduler": ["CosineAnnealingLR"],
        "grad_clip": [1.0],
        "use_relative_real_loss": [bool(USE_RELATIVE_REAL_LOSS)],
        "norm_x": ["separate(Input/theta)"],
    })
    dataframe.to_csv(os.path.join(save_path, "log.csv"), index=False)

    loss_dict = {
        "train_error": train_error,
        "test_error": test_error,
        "ET_list": ET_list,
        "best_test": np.array([best_test], dtype=np.float64),
        "best_ep": np.array([best_ep], dtype=np.int64),
        "test_part_losses_5": np.array(test_part_losses, dtype=np.float64),
        "test_order_theta": order_test.cpu().numpy(),
    }

    pred_dict = {
        "pre_test": pre_test.numpy(),
        "y_test": y_test_real_all.numpy(),
        "pre_train": pre_train.numpy(),
        "y_train": y_train_real_all.numpy(),
        "x_test_raw": x_test_raw.cpu().numpy(),
        "theta_train": theta_train.cpu().numpy(),
        "theta_test": theta_test.cpu().numpy(),
        "theta_used": theta_name,
        "theta_sorted": theta_sorted.cpu().numpy(),
    }

    sio.savemat(os.path.join(save_path, f"NORM_loss_{args.CaseName}.mat"), mdict=loss_dict)
    sio.savemat(os.path.join(save_path, f"NORM_pre_{args.CaseName}.mat"), mdict=pred_dict)

    # ==========================================================
    # ✅ NEW: Save model checkpoints
    # ==========================================================
    # 1) model.pt: BEST weights only (most commonly used)
    torch.save(best_state if best_state is not None else cpu_state_dict(model),
               os.path.join(save_path, "model.pt"))

    # 2) model_best.pt: richer info
    torch.save(
        {
            "epoch": int(best_ep),
            "best_test": float(best_test),
            "theta_used": theta_name,
            "model_state_dict": best_state if best_state is not None else cpu_state_dict(model),
            "args": dict(args.__dict__),
        },
        os.path.join(save_path, "model_best.pt")
    )

    # 3) model_last.pt: strict last weights + optimizer/scheduler (for resume)
    torch.save(
        {
            "epoch": int(epochs - 1),
            "test_last": float(test_error[-1]),
            "theta_used": theta_name,
            "model_state_dict": last_state if last_state is not None else cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": dict(args.__dict__),
        },
        os.path.join(save_path, "model_last.pt")
    )

    print(f"\nBest Testing error: {best_test:.6e} at epoch {best_ep}")
    print(f"Last Testing error: {test_error[-1]:.6e}")
    print(f"Training time: {total_time:.3f}s")
    print(f"Num of paras : {count_params(model)}")
    print(f"Saved to: {save_path}")
    print("Saved checkpoints: model.pt (best), model_best.pt, model_last.pt")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for rep in range(5):
        rep_id = rep + 1

        args = objectview({
            "modes": 128,
            "width": 32,
            "size_of_nodes": 2673,
            "batch_size": 50,
            "epochs": 1000,

            "data_dir": "../datasets/Turbulence.mat",
            "CaseName": f"Turbulence_extrap_{rep_id}",

            "lr": 0.003,

            # ✅ CHANGED: low=0, high=0.5
            "train_q_low": 0.0,
            "train_q_high": 0.5,
            "test_side": "high",     # "high" / "low" / "both"
        })

        main(args)

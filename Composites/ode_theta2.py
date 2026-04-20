import setproctitle
setproctitle.setproctitle('csq')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import time
import numpy as np
import scipy.io as sio
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_params, LpLoss, GaussianNormalizer
from rag_utils import get_rag_dataloader
from model import Approximation_block   # 你原模型依赖的 block


# ============================================================
# 1) theta from T_field: (ns, N) -> (ns,)
# ============================================================
def pick_theta_from_T_field(x: torch.Tensor, eps_std=1e-8):
    """
    x: (nsample, N)
    return theta: (nsample,), theta_name
    """
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

    # fallback: PCA PC1
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
    """
    train: [q_low, q_high] 的中间段
    test : high/low/both 外推段
    """
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


# ============================================================
# 2) Model: NORM_Net_DeltaPhi_ODE2 (theta 输入 -> coord_dim=2)
# ============================================================
class NORM_Net_DeltaPhi_ODE2(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE, steps=5, coord_dim=2):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2

        # 用 [ref_score, ref_y, grid] -> 3
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

        self.dhdt_expand = nn.Conv1d(self.width, 2 * self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        # a_input = [ref_x, x_coord]，每个都是 coord_dim -> 2*coord_dim
        self.fc3 = nn.Linear(2 * coord_dim, self.width)

        self.coord_proj = nn.Linear(coord_dim, self.width)

        self.ha_conv = nn.Conv1d(2 * self.width, self.width, 1)
        self.steps = steps
        self.coord_dim = coord_dim

        self.global_step_scale = nn.Parameter(torch.tensor(0.1))
        self.global_step_scale.data = torch.tensor(0.1)

    def func(self, a, h):
        ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
        ha = self.ha_conv(ha)           # (B, width, N)
        h = F.gelu(ha)

        x1 = self.conv0(h)
        x2 = self.w0(h)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        dhdt = x1 + x2

        dhdt = self.dhdt_expand(dhdt)  # (B, 2*width, N)
        dhdt_h, _dhdt_a = torch.split(dhdt, self.width, dim=1)
        dhdt_h = 0.5 * dhdt_h
        return dhdt_h

    def forward(self, x):
        """
        x dict from rag dataloader:
          x['x']     : (B, N, coord_dim)  -> [T, theta]
          x['ref_x'] : (B, N, coord_dim)  -> [T_ref, theta_ref]
          x['ref_y'] : (B, N) or (B, N, 1)
          x['ref_score']: (B,) or (B,1)
        """
        x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']
        B, N = x_coord.shape[0], x_coord.shape[1]

        # ref_y -> (B, N, 1)
        if ref_y.dim() == 2:
            ref_y = ref_y.unsqueeze(-1)
        else:
            ref_y = ref_y.reshape(B, N, 1)

        # ref_score -> (B, N, 1)
        ref_score = ref_score.view(B, 1, 1).repeat(1, N, 1)

        grid = self.get_grid((B, N, 1), x_coord.device)  # (B,N,1)

        # (B,N,3)
        x_in = torch.cat([ref_score, ref_y, grid], dim=-1)

        # (B,N,width)
        x_feat = self.fc0(x_in)

        # a_input: concat(ref_x, x_coord) -> (B,N,2*coord_dim)
        a_input = torch.cat([ref_x, x_coord], dim=-1)
        a_feat = self.fc3(a_input)  # (B,N,width)

        # depth_coord: (B,N,coord_dim)
        depth_coord = (x_coord - ref_x) / float(self.steps)
        depth_feat = torch.tanh(self.coord_proj(depth_coord))  # (B,N,width)

        depth_norm = torch.sigmoid(depth_feat).permute(0, 2, 1)   # (B,width,N)
        depth_scale = self.global_step_scale * depth_norm
        depth_scale = depth_scale + 1e-3

        h = x_feat.permute(0, 2, 1).contiguous()  # (B,width,N)
        a = a_feat.permute(0, 2, 1).contiguous()  # (B,width,N)

        for _ in range(self.steps):
            k1 = self.func(a, h)
            k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
            k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
            k4 = self.func(a + depth_scale, h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x_out = h.permute(0, 2, 1).contiguous()  # (B,N,width)
        x_out = F.gelu(self.fc1(x_out))
        x_out = self.fc2(x_out)                 # (B,N,1)

        return x_out + ref_y.reshape(x_out.shape)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# 3) Training script: theta 外推 + RAG
# ============================================================
def main(args):

    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PATH_data = args.data_dir
    PATH_basis = args.basis_dir

    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    modes = args.modes
    width = args.width

    step_size = 200
    gamma = 0.5

    ################################################################
    # read data & basis
    ################################################################
    data = sio.loadmat(PATH_data)
    lbo_data = sio.loadmat(PATH_basis)
    LBO_MATRIX = lbo_data['Eigenvectors']

    x_data = torch.tensor(data['T_field'], dtype=torch.float32)  # (ns, N)
    y_data = torch.tensor(data['D_field'], dtype=torch.float32)  # (ns, N) or (ns,N,1)
    if y_data.dim() == 3:
        y_data = y_data.squeeze(-1)  # -> (ns, N)

    nsample, N = x_data.shape[0], x_data.shape[1]

    ################################################################
    # theta extrap split
    ################################################################
    theta, theta_name = pick_theta_from_T_field(x_data)
    train_idx, test_idx, lo_rank, hi_rank = split_by_theta_rank(
        theta,
        train_q_low=args.train_q_low,
        train_q_high=args.train_q_high,
        test_side=args.test_side
    )

    x_train_raw = x_data[train_idx].clone()
    y_train_raw = y_data[train_idx].clone()
    x_test_raw = x_data[test_idx].clone()
    y_test_raw = y_data[test_idx].clone()

    theta_train = theta[train_idx].clone()
    theta_test = theta[test_idx].clone()

    ntrain = x_train_raw.shape[0]
    ntest = x_test_raw.shape[0]

    print(f"[Theta] name={theta_name}, std={theta.std().item():.6e}, "
          f"min={theta.min().item():.6f}, max={theta.max().item():.6f}")
    print(f"[Rank band] train ranks [{lo_rank},{hi_rank}) / {nsample}, test_side={args.test_side}")
    print(f"[Split] nsample={nsample}, ntrain={ntrain}, ntest={ntest}")

    if ntrain == 0 or ntest == 0:
        raise RuntimeError("Empty train/test after split. Adjust train_q_low/high or test_side.")

    ################################################################
    # normalization (fit on train only)
    ################################################################
    norm_x = GaussianNormalizer(x_train_raw)
    x_train_T = norm_x.encode(x_train_raw)
    x_test_T = norm_x.encode(x_test_raw)

    norm_th = GaussianNormalizer(theta_train)
    th_train = norm_th.encode(theta_train)
    th_test = norm_th.encode(theta_test)

    norm_y = GaussianNormalizer(y_train_raw)
    y_train = norm_y.encode(y_train_raw)
    y_test = norm_y.encode(y_test_raw)

    # input x: [T, theta] -> (B, N, 2)
    x_train = x_train_T.reshape(ntrain, N, 1)
    x_test = x_test_T.reshape(ntest, N, 1)

    th_train_rep = th_train.view(ntrain, 1, 1).repeat(1, N, 1)
    th_test_rep = th_test.view(ntest, 1, 1).repeat(1, N, 1)

    x_train = torch.cat([x_train, th_train_rep], dim=-1)  # (ntrain, N, 2)
    x_test = torch.cat([x_test, th_test_rep], dim=-1)     # (ntest,  N, 2)

    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_test :', x_test.shape,  'y_test :', y_test.shape)

    ################################################################
    # RAG dataloader
    ################################################################
    train_loader, test_loader = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size,
        rag_configs={"training_refer_range": 20, "refer_num": 1}
    )

    ################################################################
    # basis & model
    ################################################################
    BASE_MATRIX = LBO_MATRIX[:, :modes]
    BASE_MATRIX = torch.tensor(BASE_MATRIX, dtype=torch.float32, device=device)
    BASE_INVERSE = (BASE_MATRIX.T @ BASE_MATRIX).inverse() @ BASE_MATRIX.T

    model = NORM_Net_DeltaPhi_ODE2(
        BASE_MATRIX.shape[1],
        width,
        BASE_MATRIX,
        BASE_INVERSE,
        steps=getattr(args, "ode_steps", 5),
        coord_dim=2
    ).to(device)

    ################################################################
    # training and evaluation (with best checkpoint)
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = LpLoss(size_average=False)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))

    best_test = float("inf")
    best_ep = -1
    best_state = None

    for ep in range(epochs):
        model.train()

        train_l2_sum = 0.0
        for x, y in train_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            out = model(x)

            B = out.shape[0]
            l2 = myloss(out.view(B, -1), y.view(B, -1))
            l2.backward()
            optimizer.step()

            out_real = norm_y.decode(out.view(B, -1).detach().cpu())
            y_real = norm_y.decode(y.view(B, -1).detach().cpu())
            train_l2_sum += myloss(out_real, y_real).item()

        scheduler.step()

        model.eval()
        test_l2_sum = 0.0
        emax_sum = 0.0
        nb = 0

        with torch.no_grad():
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().to(device)
                y = y.float().to(device)

                out = model(x)
                B = out.shape[0]

                out_real = norm_y.decode(out.view(B, -1).detach().cpu())
                y_real = norm_y.decode(y.view(B, -1).detach().cpu())

                test_l2_sum += myloss(out_real, y_real).item()
                emax_sum += (out.view(B, -1) - y.view(B, -1)).abs().max(dim=1).values.mean().item()
                nb += 1

        train_l2 = train_l2_sum / ntrain
        test_l2 = test_l2_sum / ntest
        emax = emax_sum / max(nb, 1)

        train_error[ep] = train_l2
        test_error[ep] = test_l2
        ET_list[ep] = emax

        # ✅ best checkpoint
        if test_l2 < best_test:
            best_test = float(test_l2)
            best_ep = int(ep)
            best_state = cpu_state_dict(model)

        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Step: %d, Train L2: %.5f, Test L2 error: %.5f, Emax_test: %.5f, Time: %.3fs, Best: %.5f@%d'
              % (ep, train_l2, test_l2, emax, T, best_test, best_ep))
        time_step = time.perf_counter()

    print("Training done...")
    print("Best:", best_test, "@", best_ep)

    # load best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

    ################################################################
    # Predict & Save (batch_size=1)
    ################################################################
    train_loader_eval, test_loader_eval = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size=1,
        rag_configs={"training_refer_range": 20, "refer_num": 1},
        train_shuffle=False
    )

    pre_test = torch.zeros_like(y_test_raw)
    y_test_save = torch.zeros_like(y_test_raw)

    x_test_T_save = torch.zeros((ntest, N), dtype=torch.float32)
    theta_test_save = torch.zeros((ntest,), dtype=torch.float32)

    index = 0
    with torch.no_grad():
        for x, y in test_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.float().to(device)

            out = model(x)

            out_real = norm_y.decode(out.view(1, -1).detach().cpu())
            y_real = norm_y.decode(y.view(1, -1).detach().cpu())

            # x['x'] is (1, N, 2) -> [T_norm, theta_norm]
            x_in = x['x'].detach().cpu()
            T_norm = x_in[..., 0].view(1, -1)
            th_norm = x_in[:, 0, 1].view(1,)

            T_real = norm_x.decode(T_norm)
            th_real = norm_th.decode(th_norm)

            pre_test[index, :] = out_real
            y_test_save[index, :] = y_real
            x_test_T_save[index, :] = T_real.squeeze(0)
            theta_test_save[index] = th_real.squeeze(0)

            index += 1

    # ================ Save Data ====================
    current_directory = os.getcwd()
    sava_path = os.path.join(current_directory, "logs_ode", args.CaseName)
    os.makedirs(sava_path, exist_ok=True)

    train_time = float(time_step_end - time_start)

    dataframe = pd.DataFrame({
        'Test_loss_last': [float(test_error[-1])],
        'Test_loss_best': [float(best_test)],
        'Best_epoch': [int(best_ep)],
        'num_paras': [count_params(model)],
        'train_time': [train_time],
        'theta_used': [theta_name],
        'train_q_low': [args.train_q_low],
        'train_q_high': [args.train_q_high],
        'test_side': [args.test_side],
        'ntrain': [ntrain],
        'ntest': [ntest],
        'modes': [modes],
        'width': [width],
        'lr': [learning_rate],
        'ode_steps': [int(getattr(args, "ode_steps", 5))],
    })
    dataframe.to_csv(os.path.join(sava_path, 'log.csv'), index=False, sep=',')

    loss_dict = {
        'train_error': train_error,
        'test_error': test_error,
        'ET_list': ET_list,
        'best_test': np.array([best_test], dtype=np.float64),
        'best_ep': np.array([best_ep], dtype=np.int64),
    }

    pred_dict = {
        'pre_test': pre_test.cpu().detach().numpy(),
        'y_test': y_test_save.cpu().detach().numpy(),
        'x_test_T': x_test_T_save.cpu().detach().numpy(),
        'theta_test': theta_test_save.cpu().detach().numpy(),
        'theta_train': theta_train.cpu().detach().numpy(),
        'theta_name': theta_name,
    }

    sio.savemat(os.path.join(sava_path, 'NORM_loss.mat'), mdict=loss_dict)
    sio.savemat(os.path.join(sava_path, 'NORM_pre.mat'), mdict=pred_dict)

    # ✅ 保存 model.pt（包含 best 权重 + optimizer/scheduler + norm）
    ckpt = {
        "best_epoch": int(best_ep),
        "best_test_loss": float(best_test),
        "theta_used": theta_name,
        "args": dict(args.__dict__),

        "model_state_dict": best_state if best_state is not None else cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "norm_x": norm_x,
        "norm_th": norm_th,
        "norm_y": norm_y,
        "modes": int(modes),
        "width": int(width),
        "ode_steps": int(getattr(args, "ode_steps", 5)),
    }
    torch.save(ckpt, os.path.join(sava_path, "model.pt"))

    print('\nTesting error(last): %.3e' % (float(test_error[-1])))
    print('Best   error      : %.3e @ ep=%d' % (best_test, best_ep))
    print('Training time: %.3f' % train_time)
    print('Num of paras : %d' % (count_params(model)))
    print('Saved to:', sava_path)
    print('Saved checkpoint: model.pt')


if __name__ == "__main__":

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for i in range(5):
        rep = i + 1
        for args in [
            {
                'modes': 128,
                'width': 64,
                'size_of_nodes': 8232,
                'batch_size': 20,
                'epochs': 2000,
                'data_dir': '../datasets/Composites/Data/Composites.mat',
                'basis_dir': '../datasets/Composites/Composites_LBO_basis/Composites_LBO_basis.mat',
                'CaseName': 'Composite/DeltaPhiODE2_ThetaExtrap/' + str(rep),
                'lr': 0.01,

                # theta 外推
                "train_q_low": 0.2,
                "train_q_high": 0.8,
                "test_side": "high",

                # 可选：ODE steps（不写就默认 5）
                "ode_steps": 5,
            },
        ]:
            args = objectview(args)

        main(args)

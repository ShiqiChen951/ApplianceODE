import setproctitle
setproctitle.setproctitle('csq')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import numpy as np
import scipy.io as sio
import pandas as pd

import torch
import torch.nn.functional as F

from utils import count_params, LpLoss, UnitGaussianNormalizer
from model import NORM_Net_DeltaPhi_ODE2
from rag_utils import get_rag_dataloader


# ============================================================
# theta utils: from input field (ns, N) -> (ns,)
# ============================================================
def build_theta_candidates(x: torch.Tensor):
    a = x.clamp_min(1e-6)
    loga = torch.log(a)
    thetas = {
        "log_contrast": loga.max(dim=1).values - loga.min(dim=1).values,
        "loga_std": loga.std(dim=1),
        "a_std": a.std(dim=1),
        "loga_mean": loga.mean(dim=1),
        "a_mean": a.mean(dim=1),
    }
    return thetas


def pca_theta(x: torch.Tensor, max_samples_for_pca=1200):
    ns, N = x.shape
    if ns > max_samples_for_pca:
        idx = torch.randperm(ns)[:max_samples_for_pca]
        X_fit = x[idx]
    else:
        X_fit = x

    mu = X_fit.mean(dim=0, keepdim=True)
    Xc_fit = X_fit - mu

    try:
        _U, _S, V = torch.pca_lowrank(Xc_fit, q=1)
        pc1 = V[:, 0]
    except Exception:
        _U, _S, Vh = torch.linalg.svd(Xc_fit, full_matrices=False)
        pc1 = Vh[0, :]

    theta = (x - mu).matmul(pc1)
    return theta


def pick_non_degenerate_theta(x: torch.Tensor, eps_std=1e-8):
    cand = build_theta_candidates(x)
    best_name, best_theta, best_std = None, None, -1.0
    for name, th in cand.items():
        s = float(th.std().item())
        if s > best_std:
            best_std, best_name, best_theta = s, name, th

    if best_std < eps_std:
        return pca_theta(x), "pca_pc1"
    return best_theta, best_name


def make_extrapolation_split_by_rank(theta: torch.Tensor, train_frac_low=0.2, train_frac_high=0.8, test_side="high"):
    ns = theta.numel()
    order = torch.argsort(theta)  # ascending

    lo = int(round(train_frac_low * ns))
    hi = int(round(train_frac_high * ns))
    lo = max(0, min(lo, ns))
    hi = max(0, min(hi, ns))
    if hi <= lo:
        lo, hi = ns // 4, 3 * ns // 4
        if hi <= lo:
            lo, hi = 0, ns

    train_idx = order[lo:hi]
    if test_side == "high":
        test_idx = order[hi:]
    elif test_side == "low":
        test_idx = order[:lo]
    elif test_side == "both":
        test_idx = torch.cat([order[:lo], order[hi:]], dim=0)
    else:
        raise ValueError("test_side must be high / low / both")

    theta_sorted = theta[order]
    return train_idx, test_idx, lo, hi, theta_sorted


def cpu_state_dict(model: torch.nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# main
# ============================================================
def main(args):

    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PATH_data = args.data_dir
    PATH_input_basis = args.input_basis_dir
    PATH_output_basis = args.output_basis_dir

    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    modes = args.modes
    width = args.width

    step_size = 200
    gamma = 0.5

    ################################################################
    # read data
    ################################################################
    data = sio.loadmat(PATH_data)

    x_all = torch.tensor(data['input'], dtype=torch.float32)   # (ns, N)
    y_all = torch.tensor(data['output'], dtype=torch.float32)  # (ns, N)

    nsample, N = x_all.shape
    print("[Data] nsample =", nsample, "N =", N)

    ################################################################
    # theta + split (rank-based extrapolation)
    ################################################################
    theta, theta_name = pick_non_degenerate_theta(x_all)

    train_idx, test_idx, lo_rank, hi_rank, theta_sorted = make_extrapolation_split_by_rank(
        theta,
        train_frac_low=getattr(args, "train_q_low", 0.2),
        train_frac_high=getattr(args, "train_q_high", 0.8),
        test_side=getattr(args, "test_side", "high")
    )

    x_train_raw = x_all[train_idx].clone()
    y_train_raw = y_all[train_idx].clone()
    theta_train = theta[train_idx].clone()

    x_test_raw = x_all[test_idx].clone()
    y_test_raw = y_all[test_idx].clone()
    theta_test = theta[test_idx].clone()

    ntrain, ntest = x_train_raw.shape[0], x_test_raw.shape[0]
    print(f"[Split] ntrain={ntrain}, ntest={ntest}")
    print(f"[Theta] theta_name={theta_name}, std={theta.std().item():.6e}, min={theta.min().item():.6f}, max={theta.max().item():.6f}")
    print(f"[Rank band] train ranks [{lo_rank},{hi_rank}) / {nsample}, test_side={getattr(args,'test_side','high')}")

    if ntrain == 0 or ntest == 0:
        raise RuntimeError("Empty train/test after split. Adjust train_q_low/high or test_side.")

    ################################################################
    # build x=[input, theta] -> (ns, N, 2) and normalize (train-only)
    ################################################################
    theta_train_node = theta_train.view(-1, 1).repeat(1, N)
    theta_test_node = theta_test.view(-1, 1).repeat(1, N)

    x_train2 = torch.stack([x_train_raw, theta_train_node], dim=-1)  # (ntrain, N, 2)
    x_test2 = torch.stack([x_test_raw, theta_test_node], dim=-1)     # (ntest,  N, 2)

    # flatten for UnitGaussianNormalizer
    x_train_flat = x_train2.reshape(ntrain, -1)
    x_test_flat = x_test2.reshape(ntest, -1)

    norm_x = UnitGaussianNormalizer(x_train_flat)
    x_train = norm_x.encode(x_train_flat).reshape(ntrain, N, 2)
    x_test = norm_x.encode(x_test_flat).reshape(ntest, N, 2)

    norm_y = UnitGaussianNormalizer(y_train_raw)
    y_train = norm_y.encode(y_train_raw)
    y_test = norm_y.encode(y_test_raw)

    ################################################################
    # rag loader
    ################################################################
    train_loader, test_loader = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size,
        rag_configs={"training_refer_range": 20, "refer_num": 1}
    )

    ################################################################
    # read basis
    ################################################################
    LBO_Output = sio.loadmat(PATH_output_basis)['Eigenvectors']
    BASE_Output = LBO_Output[:, :modes]
    MATRIX_Output = torch.tensor(BASE_Output, dtype=torch.float32, device=device)
    INVERSE_Output = (MATRIX_Output.T @ MATRIX_Output).inverse() @ MATRIX_Output.T

    LBO_Input = sio.loadmat(PATH_input_basis)['Eigenvectors']
    BASE_Input = LBO_Input[:, :modes]
    MATRIX_Input = torch.tensor(BASE_Input, dtype=torch.float32, device=device)
    INVERSE_Input = (MATRIX_Input.T @ MATRIX_Input).inverse() @ MATRIX_Input.T

    model = NORM_Net_DeltaPhi_ODE2(
        modes, width,
        MATRIX_Output, INVERSE_Output,
        MATRIX_Input, INVERSE_Input
    ).to(device)

    ################################################################
    # training
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = LpLoss(size_average=False)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs,), dtype=np.float64)
    test_error = np.zeros((epochs,), dtype=np.float64)

    best_test = float("inf")
    best_ep = -1
    best_state = None

    for ep in range(epochs):
        model.train()
        train_l2_sum = 0.0

        for x, y in train_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            B = out.shape[0]  # ✅ 不要用固定 batch_size

            l2 = myloss(out.view(B, -1), y.view(B, -1))
            l2.backward()
            optimizer.step()

            out_real = norm_y.decode(out.view(B, -1).detach().cpu())
            y_real = norm_y.decode(y.view(B, -1).detach().cpu())
            train_l2_sum += myloss(out_real, y_real).item()

        scheduler.step()

        # eval
        model.eval()
        test_l2_sum = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().to(device)
                y = y.to(device)

                out = model(x)
                B = out.shape[0]

                out_real = norm_y.decode(out.view(B, -1).detach().cpu())
                y_real = norm_y.decode(y.view(B, -1).detach().cpu())
                test_l2_sum += myloss(out_real, y_real).item()

        train_l2 = train_l2_sum / ntrain
        test_l2 = test_l2_sum / ntest

        train_error[ep] = train_l2
        test_error[ep] = test_l2

        # ✅ best
        if test_l2 < best_test:
            best_test = float(test_l2)
            best_ep = int(ep)
            best_state = cpu_state_dict(model)

        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        print(f"Step: {ep:04d}, Train L2: {train_l2:.5f}, Test L2: {test_l2:.5f}, Time: {T:.3f}s, Best: {best_test:.5f}@{best_ep}")
        time_step = time.perf_counter()

    print("Training done...")

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

    pre_train = torch.zeros_like(y_train_raw)
    y_train_real_all = torch.zeros_like(y_train_raw)

    idx = 0
    with torch.no_grad():
        for x, y in train_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).detach().cpu())
            y_real = norm_y.decode(y.view(1, -1).detach().cpu())

            pre_train[idx, :] = out_real
            y_train_real_all[idx, :] = y_real
            idx += 1

    pre_test = torch.zeros_like(y_test_raw)
    y_test_real_all = torch.zeros_like(y_test_raw)

    idx = 0
    with torch.no_grad():
        for x, y in test_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).detach().cpu())
            y_real = norm_y.decode(y.view(1, -1).detach().cpu())

            pre_test[idx, :] = out_real
            y_test_real_all[idx, :] = y_real
            idx += 1

    # final test metric in real space
    final_test = myloss(y_test_real_all, pre_test).item() / ntest

    ################################################################
    # Save logs + mats + model.pt
    ################################################################
    current_directory = os.getcwd()
    sava_path = os.path.join(current_directory, "logs_ode", args.CaseName)
    os.makedirs(sava_path, exist_ok=True)

    total_time = float(time_step_end - time_start)

    dataframe = pd.DataFrame({
        'Test_loss_last': [float(test_error[-1])],
        'Test_loss_best': [float(best_test)],
        'Best_epoch': [int(best_ep)],
        'Test_loss_real_final': [float(final_test)],
        'Train_loss_last': [float(train_error[-1])],
        'num_paras': [count_params(model)],
        'train_time': [total_time],
        'theta_used': [theta_name],
        'train_q_low': [getattr(args, "train_q_low", 0.2)],
        'train_q_high': [getattr(args, "train_q_high", 0.8)],
        'test_side': [getattr(args, "test_side", "high")],
        'rank_lo': [int(lo_rank)],
        'rank_hi': [int(hi_rank)],
        'theta_train_min': [float(theta_train.min().item())],
        'theta_train_max': [float(theta_train.max().item())],
        'theta_test_min': [float(theta_test.min().item())],
        'theta_test_max': [float(theta_test.max().item())],
        'modes': [int(modes)],
        'width': [int(width)],
        'lr': [float(learning_rate)],
        'batch_size': [int(batch_size)],
    })
    dataframe.to_csv(os.path.join(sava_path, 'log.csv'), index=False, sep=',')

    loss_dict = {'train_error': train_error, 'test_error': test_error}

    pred_dict = {
        'pre_test': pre_test.cpu().numpy(),
        'pre_train': pre_train.cpu().numpy(),
        'y_test': y_test_real_all.cpu().numpy(),
        'y_train': y_train_real_all.cpu().numpy(),
        'theta_train': theta_train.cpu().numpy(),
        'theta_test': theta_test.cpu().numpy(),
        'theta_name': theta_name,
        'train_idx': train_idx.cpu().numpy(),
        'test_idx': test_idx.cpu().numpy(),
        'theta_sorted': theta_sorted.cpu().numpy(),
    }

    sio.savemat(os.path.join(sava_path, 'NORM_loss.mat'), mdict=loss_dict)
    sio.savemat(os.path.join(sava_path, 'NORM_pre.mat'), mdict=pred_dict)

    # ✅ model.pt (best)
    ckpt = {
        "best_epoch": int(best_ep),
        "best_test_loss": float(best_test),
        "theta_used": theta_name,
        "args": dict(args.__dict__),

        "model_state_dict": best_state if best_state is not None else cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "norm_x": norm_x,
        "norm_y": norm_y,
        "modes": int(modes),
        "width": int(width),
        "rank_lo": int(lo_rank),
        "rank_hi": int(hi_rank),
    }
    torch.save(ckpt, os.path.join(sava_path, "model.pt"))

    print('\nTesting error(real): %.3e' % final_test)
    print('Training time: %.3f' % total_time)
    print('Num of paras : %d' % count_params(model))
    print('Saved to:', sava_path)
    print('Saved checkpoint: model.pt')


if __name__ == "__main__":

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for i in range(5):
        print('====================================')
        print('NO.' + str(i) + ' repetition......')
        print('====================================')

        args = objectview({
            'modes': 128,
            'width': 64,
            'batch_size': 10,
            'epochs': 2000,
            'data_dir': '../datasets/HeatTransfer/Data/HeatTransfer.mat',
            'output_basis_dir': '../datasets/HeatTransfer/HeatTransfer_LBO_basis/lbe_ev_output.mat',
            'input_basis_dir': '../datasets/HeatTransfer/HeatTransfer_LBO_basis/lbe_ev_input.mat',
            'CaseName': 'HeatTransfer/DeltaPhiThetaExtrap/' + str(i),
            'basis': 'LBO',
            'lr': 0.01,

            # ✅ theta extrap configs (你想要 0~0.5 就改这里)
            "train_q_low": 0.0,
            "train_q_high": 0.5,
            "test_side": "high",   # high / low / both
        })

        main(args)

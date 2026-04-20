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

# 你的 Approximation_block 在 model.py 里（你原始 NORM_Net_DeltaPhi 也依赖这个）
from model import Approximation_block


# ============================================================
# 1) 从 T_field (ns, N) 提取 theta (ns,)
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


# ============================================================
# 2) 按分位数做 theta 外推 split（train_q_low/high/test_side）
# ============================================================
def split_by_theta_rank(theta: torch.Tensor, train_q_low=0.2, train_q_high=0.8, test_side="high"):
    """
    Rank-based split:
      train: middle [q_low, q_high]
      test : tail outside (high/low/both)
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
# 3) NORM_Net_DeltaPhi：输入改成 [T, theta] + RAG 参考
#    concat: x(2) + ref_score(1) + ref_x(2) + ref_y(1) + grid(1) = 7
# ============================================================
class NORM_Net_DeltaPhi(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE):
        super(NORM_Net_DeltaPhi, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2

        # 原来是 5，现在是 7（见 forward 的拼接）
        self.fc0 = nn.Linear(7, self.width)

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

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, xdict):
        """
        xdict keys from rag_utils:
          x: (B, N, 2)         -> [T, theta]
          ref_x: (B, N, 2)     -> [T_ref, theta_ref]
          ref_y: (B, N, 1) or (B, N) -> reference output
          ref_score: (B,) or (B,1)   -> scalar score
        """
        x, ref_x, ref_y, ref_score = xdict['x'], xdict['ref_x'], xdict['ref_y'], xdict['ref_score']

        # 保证 ref_y shape 为 (B, N, 1)
        if ref_y.dim() == 2:
            ref_y = ref_y.unsqueeze(-1)  # (B, N, 1)
        else:
            ref_y = ref_y.reshape(x.shape[0], x.shape[1], 1)

        B, N, _C = x.shape  # C=2

        # ref_score 做成 (B, N, 1)，不要用 ones_like(x)（否则会变成 (B,N,2)）
        if ref_score.dim() == 1:
            ref_score = ref_score.view(B, 1, 1)
        elif ref_score.dim() == 2:
            ref_score = ref_score.view(B, 1, 1)
        ref_score = ref_score.repeat(1, N, 1)  # (B, N, 1)

        grid = self.get_grid(x.shape, x.device)  # (B, N, 1)

        # concat: x(2) + ref_score(1) + ref_x(2) + ref_y(1) + grid(1) = 7
        feats = torch.cat((x, ref_score, ref_x, ref_y, grid), dim=-1)  # (B, N, 7)

        feats = self.fc0(feats)           # (B, N, width)
        feats = feats.permute(0, 2, 1)    # (B, width, N)

        x1 = self.conv0(feats)
        x2 = self.w0(feats)
        feats = F.gelu(x1 + x2)

        x1 = self.conv1(feats)
        x2 = self.w1(feats)
        feats = F.gelu(x1 + x2)

        x1 = self.conv2(feats)
        x2 = self.w2(feats)
        feats = F.gelu(x1 + x2)

        x1 = self.conv3(feats)
        x2 = self.w3(feats)
        feats = x1 + x2

        feats = feats.permute(0, 2, 1)   # (B, N, width)
        feats = F.gelu(self.fc1(feats))
        delta = self.fc2(feats)          # (B, N, 1)

        return delta + ref_y

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# ============================================================
# 4) main：theta 外推 + RAG
# ============================================================
def main(args):

    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")

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
    # reading data and reading LBO basis
    ################################################################
    data = sio.loadmat(PATH_data)
    lbo_data = sio.loadmat(PATH_basis)
    LBO_MATRIX = lbo_data['Eigenvectors']

    x_data = torch.tensor(data['T_field'], dtype=torch.float32)  # (ns, N)
    y_data = torch.tensor(data['D_field'], dtype=torch.float32)  # (ns, N) or (ns, N, 1)

    nsample, N = x_data.shape[0], x_data.shape[1]

    ################################################################
    # theta 外推 split（按分位数）
    ################################################################
    theta, theta_name = pick_theta_from_T_field(x_data)
    train_idx, test_idx, lo_rank, hi_rank = split_by_theta_rank(
        theta,
        train_q_low=args.train_q_low,
        train_q_high=args.train_q_high,
        test_side=args.test_side
    )

    x_train_raw = x_data[train_idx].clone()  # (ntrain, N)
    y_train_raw = y_data[train_idx].clone()
    x_test_raw = x_data[test_idx].clone()    # (ntest, N)
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
    # normalization（只用 train 拟合）
    ################################################################
    norm_x = GaussianNormalizer(x_train_raw)
    x_train_T = norm_x.encode(x_train_raw)
    x_test_T = norm_x.encode(x_test_raw)

    norm_th = GaussianNormalizer(theta_train)
    th_train = norm_th.encode(theta_train)   # (ntrain,)
    th_test = norm_th.encode(theta_test)     # (ntest,)

    norm_y = GaussianNormalizer(y_train_raw)
    y_train = norm_y.encode(y_train_raw)
    y_test = norm_y.encode(y_test_raw)

    # 组装输入：x = [T, theta] -> (B, N, 2)
    x_train = x_train_T.reshape(ntrain, N, 1)
    x_test = x_test_T.reshape(ntest, N, 1)

    th_train_rep = th_train.view(ntrain, 1, 1).repeat(1, N, 1)
    th_test_rep = th_test.view(ntest, 1, 1).repeat(1, N, 1)

    x_train = torch.cat([x_train, th_train_rep], dim=-1)  # (ntrain, N, 2)
    x_test = torch.cat([x_test, th_test_rep], dim=-1)     # (ntest,  N, 2)

    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_test :', x_test.shape,  'y_test :', y_test.shape)

    ################################################################
    # RAG dataloader（保持你原来的写法）
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
    BASE_MATRIX = torch.tensor(BASE_MATRIX, dtype=torch.float32).cuda()
    BASE_INVERSE = (BASE_MATRIX.T @ BASE_MATRIX).inverse() @ BASE_MATRIX.T

    model = NORM_Net_DeltaPhi(BASE_MATRIX.shape[1], width, BASE_MATRIX, BASE_INVERSE).cuda()

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = LpLoss(size_average=False)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))

    for ep in range(epochs):
        model.train()

        train_l2_sum = 0.0
        for x, y in train_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().cuda()
            y = y.float().cuda()

            optimizer.zero_grad()
            out = model(x)

            B = out.shape[0]
            l2 = myloss(out.view(B, -1), y.view(B, -1))
            l2.backward()

            out_real = norm_y.decode(out.view(B, -1).detach().cpu())
            y_real = norm_y.decode(y.view(B, -1).detach().cpu())
            train_l2_sum += myloss(out_real, y_real).item()

            optimizer.step()

        scheduler.step()

        model.eval()
        test_l2_sum = 0.0
        emax_sum = 0.0
        nb = 0

        with torch.no_grad():
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().cuda()
                y = y.float().cuda()

                out = model(x)
                B = out.shape[0]

                out_real = norm_y.decode(out.view(B, -1).cpu())
                y_real = norm_y.decode(y.view(B, -1).cpu())

                test_l2_sum += myloss(out_real, y_real).item()
                emax_sum += (out.view(B, -1) - y.view(B, -1)).abs().max(dim=1).values.mean().item()
                nb += 1

        train_l2 = train_l2_sum / ntrain
        test_l2 = test_l2_sum / ntest
        emax = emax_sum / max(nb, 1)

        train_error[ep] = train_l2
        test_error[ep] = test_l2
        ET_list[ep] = emax

        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Step: %d, Train L2: %.5f, Test L2 error: %.5f, Emax_test: %.5f, Time: %.3fs'
              % (ep, train_l2, test_l2, emax, T))
        time_step = time.perf_counter()

    print("Training done...")

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

    # 额外保存：解码后的 T 和 theta
    x_test_T_save = torch.zeros((ntest, N), dtype=torch.float32)
    theta_test_save = torch.zeros((ntest,), dtype=torch.float32)

    index = 0
    with torch.no_grad():
        for x, y in test_loader_eval:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().cuda()
            y = y.float().cuda()

            out = model(x)

            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real = norm_y.decode(y.view(1, -1).cpu())

            # x['x']: (1, N, 2) -> [T_norm, theta_norm]
            x_in = x['x'].detach().cpu()
            T_norm = x_in[..., 0].view(1, -1)
            th_norm = x_in[:, 0, 1].view(1,)  # 每个样本一个 theta

            T_real = norm_x.decode(T_norm)
            th_real = norm_th.decode(th_norm)

            pre_test[index, :] = out_real
            y_test_save[index, :] = y_real
            x_test_T_save[index, :] = T_real.squeeze(0)
            theta_test_save[index] = th_real.squeeze(0)

            index += 1

    # ================ Save Data ====================
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs_DeltaPhi/" + args.CaseName + "/"
    os.makedirs(sava_path, exist_ok=True)

    dataframe = pd.DataFrame({
        'Test_loss': [float(test_l2)],
        'num_paras': [count_params(model)],
        'train_time': [float(time_step_end - time_start)],
        'theta_used': [theta_name],
        'train_q_low': [args.train_q_low],
        'train_q_high': [args.train_q_high],
        'test_side': [args.test_side],
        'ntrain': [ntrain],
        'ntest': [ntest],
    })
    dataframe.to_csv(sava_path + 'log.csv', index=False, sep=',')

    loss_dict = {
        'train_error': train_error,
        'test_error': test_error,
        'ET_list': ET_list,
    }

    pred_dict = {
        'pre_test': pre_test.cpu().detach().numpy(),
        'y_test': y_test_save.cpu().detach().numpy(),
        'x_test_T': x_test_T_save.cpu().detach().numpy(),
        'theta_test': theta_test_save.cpu().detach().numpy(),
        'theta_train': theta_train.cpu().detach().numpy(),
        'theta_name': theta_name,
    }

    sio.savemat(sava_path + 'NORM_loss.mat', mdict=loss_dict)
    sio.savemat(sava_path + 'NORM_pre.mat', mdict=pred_dict)

    print('\nTesting error: %.3e' % (test_l2))
    print('Training time: %.3f' % (time_step_end - time_start))
    print('Num of paras : %d' % (count_params(model)))


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
                'CaseName': 'Composite/DeltaPhi_ThetaExtrap/' + str(rep),
                'lr': 0.01,

                # ===== 你要的 theta 外推配置 =====
                "train_q_low": 0.2,
                "train_q_high": 0.8,
                "test_side": "high",  # "high" / "low" / "both"
            },
        ]:
            args = objectview(args)

        main(args)

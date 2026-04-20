import setproctitle
setproctitle.setproctitle('csq')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import torch
import numpy as np
from lapy import TriaMesh, Solver
import scipy.io as sio
import time
import pandas as pd

from utils import count_params, LpLoss, GaussianNormalizer
from rag_utils import get_rag_dataloader
from model import NORM_Net_ODE2


# ================================================================
#  Rollout（保持你原逻辑不改）
# ================================================================
def rollout_evaluation(model, norm_x, norm_y, myloss, u_true_traj, T_rollout, save_path, case_name):
    model.eval()
    device = next(model.parameters()).device

    u_true_traj = u_true_traj.clone().cpu()
    T_total = u_true_traj.shape[0] - 1
    T_rollout = min(T_rollout, T_total)

    u0 = u_true_traj[0:1, :]

    x0_norm = norm_x.encode(u0)
    x0_norm = x0_norm.view(1, -1, 1).to(device)

    x_curr_norm = x0_norm

    preds = [u0.clone()]
    mse_list = []
    rel_l2_list = []

    for t in range(1, T_rollout + 1):
        B, N, _ = x_curr_norm.shape
        x_dict = {
            'x': x_curr_norm,
            'ref_x': x_curr_norm.clone(),
            'ref_y': torch.zeros_like(x_curr_norm),
            'ref_score': torch.ones(B, 1, device=device),
        }

        with torch.no_grad():
            out_norm = model(x_dict)

        out_vec_norm = out_norm.view(1, -1).cpu()

        if torch.isnan(out_vec_norm).any() or torch.isinf(out_vec_norm).any():
            print(f"[Rollout] NaN or Inf detected at step {t}, stopping rollout early.")
            break

        u_pred_t = norm_y.decode(out_vec_norm)
        preds.append(u_pred_t.clone())

        u_true_t = u_true_traj[t:t + 1, :]

        mse_t = torch.mean((u_pred_t - u_true_t) ** 2).item()

        num = myloss(u_pred_t, u_true_t).item()
        den = myloss(torch.zeros_like(u_true_t), u_true_t).item()
        rel_l2_t = num / (den + 1e-12)

        mse_list.append(mse_t)
        rel_l2_list.append(rel_l2_t)

        x_curr_norm = out_norm.detach()

    preds = torch.cat(preds, dim=0)
    mse_arr = np.array(mse_list)
    rel_l2_arr = np.array(rel_l2_list)
    T_effective = len(mse_arr)

    print(f"[Rollout] Effective rollout steps: {T_effective}/{T_rollout}")

    rollout_dict = {
        'u_true_traj': u_true_traj[:T_effective + 1, :].numpy(),
        'u_pred_traj': preds[:T_effective + 1, :].numpy(),
        'mse_t': mse_arr,
        'rel_l2_t': rel_l2_arr,
    }
    fname = f'NORM_ODE2_rollout_{case_name}.mat'
    sio.savemat(os.path.join(save_path, fname), rollout_dict)

    if T_effective > 0:
        print("\n================ Rollout Evaluation (T_effective = {}) ================".format(T_effective))
        print("Final step MSE        : {:.5e}".format(mse_arr[-1]))
        print("Final step Rel-L2     : {:.5e}".format(rel_l2_arr[-1]))
        print("Avg  over steps MSE   : {:.5e}".format(mse_arr.mean()))
        print("Avg  over steps Rel-L2: {:.5e}".format(rel_l2_arr.mean()))
        print("Rollout result saved to:", os.path.join(save_path, fname))
        print("=================================================================\n")
    else:
        print("[Rollout] No valid rollout steps (all NaN/Inf very early).")


# ================================================================
#  主流程：只保留正确的 train_loss/test_loss（物理空间 decode 后的 LpLoss）
#  + 保存 model_last.pt / model_best.pt
# ================================================================
def main(args):

    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PATH = args.data_dir

    ntrain = args.num_train
    ntest = args.num_test

    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs

    modes = args.modes
    width = args.width

    step_size = 100
    gamma = 0.5

    s = args.size_of_nodes

    # ---------------- 读数据 + LBO ----------------
    data = sio.loadmat(PATH)

    k = 128
    Points = np.vstack((data['nodes'].T, np.zeros(s).reshape(1, -1)))
    mesh = TriaMesh(Points.T, data['elements'].T - 1)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)

    # 序列数据（物理空间）
    input_all = torch.Tensor(data['Input'])    # (T, N)
    output_all = torch.Tensor(data['Output'])  # (T, N)
    u0 = input_all[0:1, :]
    u_true_traj = torch.cat([u0, output_all], dim=0)  # (T+1, N)

    x_data = input_all
    y_data = output_all

    # ---------------- 切分 ----------------
    x_train_raw = x_data[:ntrain, :]
    y_train_raw = y_data[:ntrain, :]
    x_test_raw = x_data[-ntest:, :]
    y_test_raw = y_data[-ntest:, :]

    # ---------------- 归一化 ----------------
    norm_x = GaussianNormalizer(x_train_raw)
    norm_y = GaussianNormalizer(y_train_raw)

    x_train = norm_x.encode(x_train_raw)
    x_test = norm_x.encode(x_test_raw)

    y_train = norm_y.encode(y_train_raw)
    y_test = norm_y.encode(y_test_raw)

    x_train = x_train.reshape(ntrain, -1, 1)
    x_test = x_test.reshape(ntest, -1, 1)

    rag_configs = {"training_refer_range": 20, "refer_num": 1}
    train_loader, test_loader = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size=batch_size,
        rag_configs=rag_configs
    )

    # ---------------- 模型 ----------------
    BASE_MATRIX = torch.Tensor(LBO_MATRIX[:, :modes]).to(device)
    BASE_INVERSE = (BASE_MATRIX.T @ BASE_MATRIX).inverse() @ BASE_MATRIX.T

    model = NORM_Net_ODE2(modes, width, BASE_MATRIX, BASE_INVERSE).to(device)

    # ---------------- 优化器 ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ✅ 你的 LpLoss：__call__ = rel-L2
    # 为了算 epoch mean，这里用 size_average=False 得到 sum(batch)，最后除以样本数
    myloss = LpLoss(size_average=False)

    # ---------------- 保存目录（提前建好，训练过程中可以存 best） ----------------
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, "logs_ode_real", args.CaseName)
    os.makedirs(save_path, exist_ok=True)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs,), dtype=np.float64)
    test_error = np.zeros((epochs,), dtype=np.float64)

    best_test = float("inf")
    best_epoch = -1

    # ---------------- 训练 ----------------
    for ep in range(epochs):
        model.train()
        train_sum = 0.0
        train_cnt = 0

        for x, y in train_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)

            # 反传（保持你原习惯：norm space）
            loss_back = myloss(out.view(out.shape[0], -1), y.view(y.shape[0], -1))
            loss_back.backward()
            optimizer.step()

            # ✅ 统计：物理空间 decode 后的 rel-L2（正确指标）
            out_real = norm_y.decode(out.view(out.shape[0], -1).detach().cpu())
            y_real = norm_y.decode(y.view(y.shape[0], -1).detach().cpu())
            train_sum += myloss(out_real, y_real).item()
            train_cnt += out.shape[0]

        scheduler.step()

        # ---------------- 测试 ----------------
        model.eval()
        test_sum = 0.0
        test_cnt = 0

        with torch.no_grad():
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().to(device)
                y = y.to(device)

                out = model(x)
                out_real = norm_y.decode(out.view(out.shape[0], -1).detach().cpu())
                y_real = norm_y.decode(y.view(y.shape[0], -1).detach().cpu())
                test_sum += myloss(out_real, y_real).item()
                test_cnt += out.shape[0]

        train_l2 = train_sum / max(train_cnt, 1)
        test_l2 = test_sum / max(test_cnt, 1)

        train_error[ep] = train_l2
        test_error[ep] = test_l2

        # ---------------- 保存 best checkpoint（按 test_loss 最小） ----------------
        if test_l2 < best_test:
            best_test = float(test_l2)
            best_epoch = int(ep)
            best_ckpt_path = os.path.join(save_path, "model_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_test_loss": best_test,
                    "best_epoch": best_epoch,
                    "epoch_train_loss": train_error,
                    "epoch_test_loss": test_error,
                    "num_params": count_params(model),
                    "args": args.__dict__ if hasattr(args, "__dict__") else dict(args),
                },
                best_ckpt_path
            )

        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        print(f"Epoch: {ep}, Train loss: {train_l2:.6e}, Test loss: {test_l2:.6e}, Time: {T:.3f}s")
        time_step = time.perf_counter()

    time_end = time.perf_counter()

    print("\n=============================")
    print("Training done...")
    print("=============================\n")

    # ---------------- 保存 last checkpoint ----------------
    last_ckpt_path = os.path.join(save_path, "model_last.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch_train_loss": train_error,
            "epoch_test_loss": test_error,
            "num_params": count_params(model),
            "train_time_sec": float(time_end - time_start),
            "args": args.__dict__ if hasattr(args, "__dict__") else dict(args),
        },
        last_ckpt_path
    )

    # ---------------- 保存 epoch loss 曲线 ----------------
    df_epoch = pd.DataFrame({
        "epoch": np.arange(epochs),
        "train_loss": train_error,
        "test_loss": test_error
    })
    df_epoch.to_csv(os.path.join(save_path, "epoch_loss.csv"), index=False)

    # 你原来的 log.csv（保留）
    dataframe = pd.DataFrame({
        "Test_loss": [float(test_error[-1])],
        "num_paras": [count_params(model)],
        "train_time": [float(time_end - time_start)]
    })
    dataframe.to_csv(os.path.join(save_path, "log.csv"), index=False, sep=",")

    print("Saved last model to:", last_ckpt_path)
    print("Saved best model to:", os.path.join(save_path, "model_best.pt"))
    print("Best test loss:", best_test, "at epoch", best_epoch)

    print('\nTesting error: %.3e' % (float(test_error[-1])))
    print('Training time: %.3f' % (float(time_end - time_start)))
    print('Num of paras : %d' % (count_params(model)))

    # ---------------- Rollout（仍然保留）----------------
    T_rollout = 200
    rollout_evaluation(
        model=model,
        norm_x=norm_x,
        norm_y=norm_y,
        myloss=myloss,
        u_true_traj=u_true_traj,
        T_rollout=T_rollout,
        save_path=save_path,
        case_name=args.CaseName
    )

    # ---------------- 释放显存（避免循环跑 5 次时 busy）----------------
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================================================
# entry
# ================================================================
if __name__ == "__main__":

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for i in range(5):
        ii = i + 1
        for args in [
            {
                'modes': 128,
                'width': 32,
                'size_of_nodes': 2673,
                'batch_size': 50,
                'epochs': 1000,
                'data_dir': '../datasets/Turbulence',
                'num_train': 300,
                'num_test': 100,
                'CaseName': 'Turbulence_' + str(ii),
                'lr': 0.01
            },
        ]:
            args = objectview(args)
            main(args)

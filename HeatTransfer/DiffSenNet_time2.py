import setproctitle
setproctitle.setproctitle('csq')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import time
import pandas as pd

from utils import count_params, LpLoss, UnitGaussianNormalizer
from model import NORM_Net_DeltaPhi_ODE2
from rag_utils import get_rag_dataloader


# ================================================================
#  Rollout 评估：基于已解码好的 y_true / y_pred 序列（非自回归）
# ================================================================
def rollout_evaluation(
        myloss,
        y_true_seq,    # (T_total, N) 物理空间
        y_pred_seq,    # (T_total, N) 物理空间
        save_path,
        case_name,
        T_rollout,
        test_l2_scalar=None
    ):

    y_true_seq = y_true_seq.clone().cpu()
    y_pred_seq = y_pred_seq.clone().cpu()
    assert y_true_seq.shape == y_pred_seq.shape, "y_true_seq 和 y_pred_seq 形状必须一致"

    T_total, N = y_true_seq.shape
    T_effective = min(T_rollout, T_total)

    mse_list = []
    rel_l2_list = []

    for t in range(T_effective):
        u_true_t = y_true_seq[t:t+1, :]
        u_pred_t = y_pred_seq[t:t+1, :]

        mse_t = torch.mean((u_pred_t - u_true_t) ** 2).item()

        num = myloss(u_pred_t, u_true_t).item()
        den = myloss(torch.zeros_like(u_true_t), u_true_t).item()
        rel_l2_t = num / (den + 1e-12)

        mse_list.append(mse_t)
        rel_l2_list.append(rel_l2_t)

    mse_arr = np.array(mse_list)
    rel_l2_arr = np.array(rel_l2_list)

    os.makedirs(save_path, exist_ok=True)
    safe_case_name = str(case_name).replace('/', '_').replace('\\', '_')
    filename = f'NORM_rollout_{safe_case_name}.mat'

    rollout_dict = {
        'u_true_traj': y_true_seq[:T_effective].numpy(),
        'u_pred_traj': y_pred_seq[:T_effective].numpy(),
        'mse_t': mse_arr,
        'rel_l2_t': rel_l2_arr,
        'test_l2': np.array(test_l2_scalar if test_l2_scalar is not None else np.nan),
    }
    sio.savemat(os.path.join(save_path, filename), rollout_dict)

    print(f"\n================ Rollout Evaluation ================")
    print(f"T_total     = {T_total}")
    print(f"T_rollout   = {T_rollout} (user-defined)")
    print(f"T_effective = {T_effective} (actually evaluated)")
    print("Final step MSE        : {:.5e}".format(mse_arr[-1]))
    print("Final step Rel-L2     : {:.5e}".format(rel_l2_arr[-1]))
    print("Avg  over steps MSE   : {:.5e}".format(mse_arr.mean()))
    print("Avg  over steps Rel-L2: {:.5e}".format(rel_l2_arr.mean()))
    print("Rollout result saved to:", os.path.join(save_path, filename))
    print("====================================================\n")


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

    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    modes = args.modes
    width = args.width

    step_size = 200
    gamma = 0.5

    # ---------------- save dir（提前建好，训练中保存 best / last） ----------------
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, "logs_ode", args.CaseName)
    os.makedirs(save_path, exist_ok=True)
    print("save_path:", save_path)

    # ---------------- read data ----------------
    data = sio.loadmat(PATH_data)

    x_train = torch.Tensor(data['input'][0:ntrain])
    x_test = torch.Tensor(data['input'][-ntest:])

    y_train = torch.Tensor(data['output'][0:ntrain])
    y_test = torch.Tensor(data['output'][-ntest:])

    # ---------------- normalization ----------------
    norm_x = UnitGaussianNormalizer(x_train)
    norm_y = UnitGaussianNormalizer(y_train)

    x_train = norm_x.encode(x_train)
    x_test = norm_x.encode(x_test)

    y_train = norm_y.encode(y_train)
    y_test = norm_y.encode(y_test)

    x_train = x_train.reshape(ntrain, -1, 1)
    x_test = x_test.reshape(ntest, -1, 1)

    # ---------------- RAG dataloader ----------------
    rag_configs = {"training_refer_range": 20, "refer_num": 1}
    train_loader, test_loader = get_rag_dataloader(
        x_train, y_train, x_test, train_y=y_test,  # 兼容你可能的 rag_utils 形参
        batch_size=batch_size,
        rag_configs=rag_configs
    ) if "train_y" in get_rag_dataloader.__code__.co_varnames else get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size=batch_size,
        rag_configs=rag_configs
    )

    # ---------------- read basis ----------------
    LBO_Output = sio.loadmat(PATH_output_basis)['Eigenvectors']
    BASE_Output = LBO_Output[:, :modes]
    MATRIX_Output = torch.Tensor(BASE_Output).to(device)
    INVERSE_Output = (MATRIX_Output.T @ MATRIX_Output).inverse() @ MATRIX_Output.T

    LBO_Input = sio.loadmat(PATH_input_basis)['Eigenvectors']
    BASE_Input = LBO_Input[:, :modes]
    MATRIX_Input = torch.Tensor(BASE_Input).to(device)
    INVERSE_Input = (MATRIX_Input.T @ MATRIX_Input).inverse() @ MATRIX_Input.T

    model = NORM_Net_DeltaPhi_ODE2(
        modes, width,
        MATRIX_Output, INVERSE_Output,
        MATRIX_Input, INVERSE_Input
    ).to(device)

    # ---------------- optim ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ✅ 只用 LpLoss(rel)（你给的实现）
    myloss = LpLoss(size_average=False)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    train_error = np.zeros((epochs,), dtype=np.float64)
    test_error = np.zeros((epochs,), dtype=np.float64)

    best_test = float("inf")
    best_epoch = -1

    # ---------------- train ----------------
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

            # backward 在 norm space（保持你原逻辑）
            loss_back = myloss(out.view(out.shape[0], -1), y.view(y.shape[0], -1))
            loss_back.backward()
            optimizer.step()

            # ✅ 统计：decode 到物理空间后的 LpLoss(rel)
            out_real = norm_y.decode(out.view(out.shape[0], -1).detach().cpu())
            y_real = norm_y.decode(y.view(y.shape[0], -1).detach().cpu())
            train_sum += myloss(out_real, y_real).item()
            train_cnt += out.shape[0]

        scheduler.step()

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

        # ✅ 每个 epoch 的 train/test：按样本平均（不是除以 ntrain/ntest 的“累计和”）
        train_l2 = train_sum / max(train_cnt, 1)
        test_l2 = test_sum / max(test_cnt, 1)

        train_error[ep] = train_l2
        test_error[ep] = test_l2

        # ✅ 保存 best
        if test_l2 < best_test:
            best_test = float(test_l2)
            best_epoch = int(ep)
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
                os.path.join(save_path, "model_best.pt")
            )

        time_step_end = time.perf_counter()
        T_cost = time_step_end - time_step
        print(f"Epoch: {ep}, Train loss: {train_l2:.6e}, Test loss: {test_l2:.6e}, Time: {T_cost:.3f}s")
        time_step = time.perf_counter()

    time_end = time.perf_counter()
    print("Training done...")

    # ✅ 保存 last
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
        os.path.join(save_path, "model_last.pt")
    )

    # 保存 epoch loss
    df_epoch = pd.DataFrame({
        "epoch": np.arange(epochs),
        "train_loss": train_error,
        "test_loss": test_error
    })
    df_epoch.to_csv(os.path.join(save_path, "epoch_loss.csv"), index=False)

    # ========= batch_size=1 推理，得到物理空间 y_test / pre_test =========
    train_loader_1, test_loader_1 = get_rag_dataloader(
        x_train, y_train, x_test, y_test,
        batch_size=1,
        rag_configs=rag_configs,
        train_shuffle=False
    )

    pre_test = torch.zeros((ntest, y_test.shape[-1]), dtype=torch.float32)
    y_test_real = torch.zeros((ntest, y_test.shape[-1]), dtype=torch.float32)

    idx = 0
    with torch.no_grad():
        for x, y in test_loader_1:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().to(device)
            y = y.to(device)

            out = model(x)

            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real = norm_y.decode(y.view(1, -1).cpu())

            pre_test[idx, :] = out_real
            y_test_real[idx, :] = y_real
            idx += 1

    # 保存 log / mat（保持你原风格）
    dataframe = pd.DataFrame({
        'Test_loss': [float(test_error[-1])],
        'Train_loss': [float(train_error[-1])],
        'num_paras': [count_params(model)],
        'train_time': [float(time_end - time_start)]
    })
    dataframe.to_csv(os.path.join(save_path, 'log.csv'), index=False, sep=',')

    loss_dict = {'train_error': train_error, 'test_error': test_error}
    pred_dict = {
        'pre_test': pre_test.numpy(),
        'y_test': y_test_real.numpy(),
    }

    sio.savemat(os.path.join(save_path, 'NORM_loss.mat'), mdict=loss_dict)
    sio.savemat(os.path.join(save_path, 'NORM_pre.mat'), mdict=pred_dict)

    print('\nFinal Test loss: %.3e' % (float(test_error[-1])))
    print('Training time: %.3f' % (float(time_end - time_start)))
    print('Num of paras : %d' % (count_params(model)))
    print("Saved last model to:", os.path.join(save_path, "model_last.pt"))
    print("Saved best model to:", os.path.join(save_path, "model_best.pt"))
    print("Best test loss:", best_test, "at epoch", best_epoch)

    # ================ Rollout 评估：使用用户设定的 T_rollout ====================
    T_rollout = getattr(args, "T_rollout", y_test_real.shape[0])

    rollout_evaluation(
        myloss=myloss,
        y_true_seq=y_test_real,   # 物理空间真值
        y_pred_seq=pre_test,      # 物理空间预测
        save_path=save_path,
        case_name=args.CaseName,
        T_rollout=T_rollout
    )

    # 释放显存（循环跑多次防 busy）
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for i in range(5):
        print('====================================')
        print('NO.' + str(i) + ' repetition......')
        print('====================================')

        for args in [
            {
                'modes': 128,
                'width': 64,
                'batch_size': 10,
                'epochs': 2000,
                'data_dir': '../datasets/HeatTransfer/Data/HeatTransfer.mat',
                'output_basis_dir': '../datasets/HeatTransfer/HeatTransfer_LBO_basis/lbe_ev_output.mat',
                'input_basis_dir': '../datasets/HeatTransfer/HeatTransfer_LBO_basis/lbe_ev_input.mat',
                'num_train': 100,
                'num_test': 100,
                'CaseName': 'HeatTransfer/DeltaPhi/' + str(i),
                'basis': 'LBO',
                'lr': 0.01,
                'T_rollout': 50,
            },
        ]:
            args = objectview(args)

        main(args)

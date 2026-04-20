import setproctitle
setproctitle.setproctitle('csq')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
import scipy.io as sio
import time
import pandas as pd
from utils import count_params,LpLoss, GaussianNormalizer
from model import NORM_net_DeltaPhi_ODE2
from rag_utils import get_rag_dataloader

from Adam import Adam

def rollout_evaluation_time(myloss,
                            y_true_seq,   # (Ntest, Nt, Nnodes, 3)，物理空间
                            y_pred_seq,   # (Ntest, Nt, Nnodes, 3)
                            save_path,
                            case_name,
                            T_rollout,
                            test_l2_scalar=None):   # 用户设定的“预测时间步数”
    """
    针对 BloodFlow 序列数据的“时间步误差评估”（非自回归）：

    - y_true_seq / y_pred_seq: (Ntest, Nt, Nnodes, 3)
      Ntest : 测试样本数
      Nt    : 每个样本的总时间步
      Nnodes: 空间节点数
      3     : 速度分量 (vx, vy, vz)

    - 只在前 T_rollout 个时间步上统计误差：
        T_effective = min(T_rollout, Nt)
    """

    # 确保在 CPU 上
    y_true_seq = y_true_seq.clone().cpu()
    y_pred_seq = y_pred_seq.clone().cpu()

    assert y_true_seq.shape == y_pred_seq.shape, "y_true_seq 和 y_pred_seq 形状必须一致"

    Ntest, Nt, Nnodes, C = y_true_seq.shape # Nt:1656
    T_effective = min(T_rollout, Nt)

    mse_list = []
    rel_l2_list = []

    for t in range(T_effective):
        # 所有样本在第 t 个时间步的场 (Ntest, Nnodes, 3) → 展平成 (Ntest, Nnodes*3)
        u_true_t = y_true_seq[:, t, :, :].reshape(Ntest, -1)
        u_pred_t = y_pred_seq[:, t, :, :].reshape(Ntest, -1)

        # MSE（batch + 空间平均）
        mse_t = torch.mean((u_pred_t - u_true_t) ** 2).item()

        # 相对 L2
        num = myloss(u_pred_t, u_true_t).item()
        den = myloss(torch.zeros_like(u_true_t), u_true_t).item()
        rel_l2_t = num / (den + 1e-12)

        mse_list.append(mse_t)
        rel_l2_list.append(rel_l2_t)

    mse_arr = np.array(mse_list)           # (T_effective,)
    rel_l2_arr = np.array(rel_l2_list)

    os.makedirs(save_path, exist_ok=True)
    safe_case_name = str(case_name).replace('/', '_').replace('\\', '_')
    filename = f"NORM_rollout_{safe_case_name}_T{T_effective}.mat"

    # 时间维放前面：(T_effective, Ntest, Nnodes, 3)
    u_true_traj = y_true_seq[:, :T_effective].permute(1, 0, 2, 3).numpy()
    u_pred_traj = y_pred_seq[:, :T_effective].permute(1, 0, 2, 3).numpy()

    rollout_dict = {
        "u_true_traj": u_true_traj,
        "u_pred_traj": u_pred_traj,
        "mse_t": mse_arr,
        "rel_l2_t": rel_l2_arr,
        "test_l2": np.array(test_l2_scalar if test_l2_scalar is not None else np.nan),
    }
    sio.savemat(os.path.join(save_path, filename), rollout_dict)

    print("\n================ Time-wise Rollout Evaluation ================")
    print(f"Nt total      = {Nt}")
    print(f"T_rollout req = {T_rollout}")
    print(f"T_effective   = {T_effective}")
    print("Final step MSE        : {:.5e}".format(mse_arr[-1]))
    print("Final step Rel-L2     : {:.5e}".format(rel_l2_arr[-1]))
    print("Avg  over steps MSE   : {:.5e}".format(mse_arr.mean()))
    print("Avg  over steps Rel-L2: {:.5e}".format(rel_l2_arr.mean()))
    print("Rollout result saved to:", os.path.join(save_path, filename))
    print("==============================================================\n")

def main(args):  
    
    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")
    
    LBO_PATH = args.LBO_dir
    PATH = args.data_dir
    
    ntrain = args.num_train  
    ntest = args.num_test  
    batch_size = args.batch_size 
    learning_rate = args.lr    
    epochs = args.epochs    
    modes = args.modes 
    Fmodes = args.Fmodes 
    width = args.width 
    
    nodes = args.size_of_nodes
    BASIS = args.basis 
    
    step_size = 100
    gamma = 0.1
    
    ################################################################
    # reading data reading LBO basis
    ################################################################   
    
    data = sio.loadmat(PATH) 
    LBOdata = sio.loadmat(LBO_PATH) 
    LBO_MATRIX = LBOdata['Eigenvectors']
    
    x_dataIn = torch.Tensor(data['BC_time'])
    y_dataIn1 = torch.Tensor(data['velocity_x'])
    y_dataIn2 = torch.Tensor(data['velocity_y'])
    y_dataIn3 = torch.Tensor(data['velocity_z'])
    
    x_data = x_dataIn
    y_data = torch.zeros((y_dataIn1.shape[0],y_dataIn1.shape[1],y_dataIn1.shape[2],3))
     
    y_data[:,:,:,0] = y_dataIn1
    y_data[:,:,:,1] = y_dataIn2
    y_data[:,:,:,2] = y_dataIn3
    
    ################################################################
    # normalization
    ################################################################  
    x_train = x_data[:ntrain,:,:]
    y_train = y_data[:ntrain,:,:]
    x_test = x_data[-ntest:,:,:]
    y_test = y_data[-ntest:,:,:]
            

    norm_x1 = GaussianNormalizer(x_train[:,:,0])
    norm_x2 = GaussianNormalizer(x_train[:,:,1:])
    
    x_train[:,:,0] = norm_x1.encode(x_train[:,:,0])
    x_train[:,:,1:] = norm_x2.encode(x_train[:,:,1:])
    x_test[:,:,0] = norm_x1.encode(x_test[:,:,0])
    x_test[:,:,1:] = norm_x2.encode(x_test[:,:,1:])
    
    norm_y  = GaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
       
    train_loader, test_loader = get_rag_dataloader(x_train, y_train, x_test, y_test,  batch_size, rag_configs = { "training_refer_range": 20, "refer_num": 1 } )

    if BASIS == 'LBO':
        BASE_MATRIX = LBO_MATRIX[:,:modes] 

    TIME_MATRIX = BASE_MATRIX
    
    TIME_MATRIX = torch.Tensor(TIME_MATRIX).to(device) 
    TIME_INVERSE = (TIME_MATRIX.T@TIME_MATRIX).inverse()@TIME_MATRIX.T 
        
    BASE_MATRIX = torch.Tensor(BASE_MATRIX).to(device) 
    BASE_INVERSE = (BASE_MATRIX.T@BASE_MATRIX).inverse()@BASE_MATRIX.T 
    
    Nt = x_train.shape[1]
    
    model = NORM_net_DeltaPhi_ODE2(modes, nodes, Fmodes, width, TIME_MATRIX, TIME_INVERSE, BASE_MATRIX, BASE_INVERSE, Nt).to(device) 
    
    ################################################################
    # training and evaluation
    ################################################################

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
    
    myloss = LpLoss(d=3, p=2, size_average  = False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    
    for ep in range(epochs):
        
        model.train() 
        train_l2 = 0
        for x, y in train_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().cuda()
            y = y.cuda()  
    
            optimizer.zero_grad()
            out = model(x)
            # y = y[:, -1, :, :]   # → [B, N, C]  
            # print(out.shape,y.shape) 
            # torch.Size([10, 121, 3]) torch.Size([10, 1656, 121, 3])
            # exit()      
            # y_fft = torch.fft.rfft(y, dim=1)
            # y_low = y_fft[:, :121]         # 保留前 121 个频率
            # y = torch.fft.irfft(y_low, n=121, dim=1)
            # y = y.mean(dim=1)   # [10, 121, 3]
            # print(out.shape, y.shape)
            l2 = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))            
            l2.backward() 
                        
            out_real = norm_y.decode(out.cpu()).contiguous().reshape(batch_size, -1)
            y_real = norm_y.decode(y.cpu()).reshape(batch_size, -1) 
            train_l2 += myloss(out_real, y_real).item()          
                   
            optimizer.step()
            
        scheduler.step()
        model.eval() 
        test_l2 = 0.0
        
        test_l2_eval = 0.0
        with torch.no_grad(): 
            for x, y in test_loader:
                for keyname in x.keys():
                    x[keyname] = x[keyname].float().cuda()
                y = y.cuda()  
    
                out = model(x) 
                # print(out.shape, y.shape)
                # y = y[:, -1, :, :]
                # y_fft = torch.fft.rfft(y, dim=1)
                # y_low = y_fft[:, :121]         # 保留前 121 个频率
                # y = torch.fft.irfft(y_low, n=121, dim=1)
                # y = y.mean(dim=1)   # [10, 121, 3]
                out_real = norm_y.decode(out.cpu()).contiguous().reshape(batch_size, -1)
                y_real = norm_y.decode(y.cpu()).reshape(batch_size, -1)
                test_l2 += myloss(out_real, y_real).item()                
                loss_max_test= (abs(out_real- y_real)).max(axis=1).values.mean()
                test_l2_eval += myloss(out_real.reshape(1, -1), y_real.reshape(1, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest
        test_l2_eval /= ntest
        train_error[ep] = train_l2
        test_error[ep] = test_l2
        
        ET_list[ep] = loss_max_test
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Epoch: %d, Train L2: %.5f, Test L2: %.5f, Emax_te: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, loss_max_test, T))
        time_step = time.perf_counter()
          
    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    
    train_loader, test_loader = get_rag_dataloader(x_train, y_train, x_test, y_test,  batch_size = 1, rag_configs = { "training_refer_range": 20, "refer_num": 1 }, train_shuffle=False )

    # pre_test = torch.zeros(y_test.shape)     
    # y_test   = torch.zeros(y_test.shape)      
    # x_test   = torch.zeros(x_test.shape)      
    pre_test = torch.zeros_like(y_test)
    y_test_real_buf = torch.zeros_like(y_test)   # 这个专门存 decode 后的真实 y
    x_test_buf = torch.zeros_like(x_test)

    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            for keyname in x.keys():
                x[keyname] = x[keyname].float().cuda()
            y = y.cuda()  
            
            out = model(x)
            
            out_real = norm_y.decode(out.cpu())
            y_real   = norm_y.decode(y.cpu())
            
            x_real   = x
            # x_real[:,:,0] = norm_x1.decode(x_real[:,:,0].cpu())
            x_real["x"][:, :, 0] = norm_x1.decode(
                x_real["x"][:, :, 0].cpu()
            ).to(x_real["x"].device)
            # x_real[:,:,1:] = norm_x2.decode(x_real[:,:,1:].cpu())
            x_real["x"][:, :, 1:] = norm_x2.decode(
                x_real["x"][:, :, 1:].cpu()
            ).to(x_real["x"].device)
            
            pre_test[index,:] = out_real
            y_test_real_buf[index,:] = y_real
            x_test_buf[index,:] = x_real["x"]
            
            index = index + 1
            
    # ================ Save Data ====================
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs_DeltaPhi_ode/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
    
    dataframe = pd.DataFrame({'Test_loss': [test_l2],
                              'num_paras': [count_params(model)],
                              'train_time':[time_step_end - time_start]})
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error}
    
    pred_dict = {'pre_test'   : pre_test.cpu().detach().numpy(),
                    'x_test'  : x_test.cpu().detach().numpy(),
                    'y_test'  : y_test.cpu().detach().numpy(),
                    }
    
    # sio.savemat(sava_path +'NORM_loss_' + args.CaseName + '.mat', mdict = loss_dict)                                                     
    # sio.savemat(sava_path +'NORM_pre_'  + args.CaseName + '.mat', mdict = pred_dict)
    

    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))
        # ================ 时间方向 rollout 评估（用户设定时间步数） =================
    # 若没有在 args 里显式设定，则默认用完整时间长度
    Nt_total = y_test.shape[1]
    T_rollout = getattr(args, "T_rollout", Nt_total)

    rollout_evaluation_time(
        myloss=myloss,
        y_true_seq=y_test,      # (Ntest, Nt, Nnodes, 3)
        y_pred_seq=pre_test,    # (Ntest, Nt, Nnodes, 3)
        save_path=sava_path,
        case_name=args.CaseName,
        test_l2_scalar=test_l2_eval,
        T_rollout=T_rollout
    )  
    

if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            

    for i in range(5):
        
        i = i + 1
        for args in [
            { 'modes': 64,  
              'Fmodes': 16,
              'width': 16,
              'size_of_nodes': 1656,
              'batch_size': 10, 
              'epochs': 500,
              'data_dir': '../datasets/BloodFlow',
              'LBO_dir': '../datasets/BloodFlow_LBO_basis/LBO_basis',
              'num_train': 400, 
              'num_test': 100,
              'CaseName': 'velocity_xyz_'+str(i),
              'basis':'LBO',
              'lr' : 0.001,
              'T_rollout': 100},
        ]:
            args = objectview(args)
    
        main(args)


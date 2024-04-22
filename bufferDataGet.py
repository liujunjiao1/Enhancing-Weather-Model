import numpy as np
import os
import torch

def get_rehearsal_set(numpy_folder, indexs):
    x_flag = True
    for i in indexs:
        x_data = np.load(os.path.join(numpy_folder, str(i) + '.npy'))
        x_data = np.expand_dims(x_data,0)
        y_flag = True
        for j in range(5):
            y_data_sample = np.load(os.path.join(numpy_folder, str(i+j+1) + '.npy'))
            mx2t_data = y_data_sample[2, 7, :, :]
            mx2t_data = np.expand_dims(mx2t_data, 0)
            mx2t_data = np.expand_dims(mx2t_data, 0)
            mn2t_data = y_data_sample[2, 10, :, :]  # 8:72
            mn2t_data = np.expand_dims(mn2t_data, 0)
            mn2t_data = np.expand_dims(mn2t_data, 0)
            y_data_sample = np.concatenate((mx2t_data, mn2t_data), axis=1)
            # print(y_data_sample.shape)
            if y_flag:
                y_flag = False
                y_data = y_data_sample
            else:
                y_data = np.concatenate((y_data_sample, y_data),axis=0)
        y_data = np.expand_dims(y_data,0)
        if x_flag:
            x_flag = False
            seq_x = x_data
            seq_y = y_data
        else:
            seq_x = np.concatenate((x_data,seq_x), axis=0)
            seq_y = np.concatenate((y_data,seq_y), axis=0)

    input = torch.from_numpy(seq_x).contiguous().float()  # 返回内存中连续的tensor
    output = torch.from_numpy(seq_y).contiguous().float()
    return input, output

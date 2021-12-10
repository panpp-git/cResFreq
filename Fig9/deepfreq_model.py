
import numpy as np
import torch
import util
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio

# In[2]:
def myModel():
    fr_path = 'deepfreq_epoch_120.pth'
    xgrid = np.linspace(-0.5, 0.5, 4096, endpoint=False)
    # load models
    fr_module, _, _, _, _ = util.load(fr_path, 'fr')
    fr_module.cpu()
    fr_module.eval()

    f = h5py.File('matlab_real1.h5', 'r')
    real_data = f['matlab_real1'][:]
    f.close()
    f = h5py.File('matlab_imag1.h5', 'r')
    imag_data = f['matlab_imag1'][:]
    f.close()
    bz=1;
    N = 64

    signal_50dB = np.zeros([int(bz), 2, N]).astype(np.float32)
    signal_50dB[:, 0,:] = (real_data.astype(np.float32)).T
    signal_50dB[:, 1,:] = (imag_data.astype(np.float32)).T
    signal_50dB_c = signal_50dB[:, 0] + 1j * signal_50dB[:, 1]

    with torch.no_grad():
        fr_50dB = fr_module(torch.tensor(signal_50dB))
        fr_50dB = fr_50dB.cpu().data.numpy()

        dataNew = 'data1_deepfreq.mat'
        sio.savemat(dataNew, {'data1_deepfreq':fr_50dB})

myModel()
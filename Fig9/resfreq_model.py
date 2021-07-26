import os
import torch
import util
import numpy as np
import fr
import h5py
import scipy.io as sio

# In[2]:
def myModel():
    fr_path = 'cResFreq_epoch_60.pth'
    xgrid = np.linspace(-0.5, 0.5, 4096, endpoint=False)
    # load models
    fr_module, _, _, _, _ = util.load(fr_path, 'skip')
    fr_module.cpu()
    fr_module.eval()

    f = h5py.File('matlab_real2.h5', 'r')
    real_data2 = f['matlab_real2'][:]
    f.close()
    f = h5py.File('matlab_imag2.h5', 'r')
    imag_data2 = f['matlab_imag2'][:]
    f.close()
    bz=1;
    N = 64


    signal_50dB2 = np.zeros([int(bz), 2, N]).astype(np.float32)
    signal_50dB2[:, 0,:] = (real_data2.astype(np.float32)).T
    signal_50dB2[:, 1,:] = (imag_data2.astype(np.float32)).T


    with torch.no_grad():

        fr_50dB2 = fr_module(torch.tensor(signal_50dB2))
        fr_50dB2 = fr_50dB2.cpu().data.numpy()

        dataNew = 'data1_resfreq.mat'
        sio.savemat(dataNew, {'data1_resfreq':fr_50dB2})



myModel()
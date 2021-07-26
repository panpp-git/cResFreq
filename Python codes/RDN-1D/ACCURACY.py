
import argparse
import numpy as np
from data import noise
from data.data import gen_signal_accuracy
import torch
import util
from numpy.fft import fft,fftshift
import pickle
from data import fr
import matplotlib.pyplot as plt
import matlab.engine
import h5py
eng = matlab.engine.start_matlab()

def crlb(N,SNR):
    k=np.array(list(range(-1*int(N/2),int((N-1)/2))))
    a0=10**(SNR/20)
    l1=k**2
    J=8*np.pi*np.pi*a0*a0*np.sum(l1)
    fc=np.sqrt(1/J)
    return fc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='./test_accuracy', type=str,
                        help="The output directory where the data will be written.")
    parser.add_argument('--overwrite', action='store_true',default=1,
                        help="Overwrite the content of the output directory")

    parser.add_argument("--n_test", default=1, type=int,
                        help="Number of signals")
    parser.add_argument("--signal_dimension", default=64, type=int,
                        help="Dimension of sinusoidal signal")
    parser.add_argument("--minimum_separation", default=0.5, type=float,
                        help="Minimum distance between spikes, normalized by 1/signal_dim")
    parser.add_argument("--max_freq", default=10, type=int,
                        help="Maximum number of frequency, the distribution is uniform between 1 and max_freq")
    parser.add_argument("--distance", default="normal", type=str,
                        help="Distribution type of the inter-frequency distance")
    parser.add_argument("--amplitude", default="normal_floor", type=str,
                        help="Distribution type of the spike amplitude")
    parser.add_argument("--floor_amplitude", default=0.1, type=float,
                        help="Minimum spike amplitude (only used for the normal_floor distribution)")
    parser.add_argument('--dB', nargs='+', default=['-15', '-10','-5', '0', '5', '10', '15', '20', '25', '30'],
                        help='additional dB levels')

    parser.add_argument("--numpy_seed", default=105, type=int,
                        help="Numpy seed")
    parser.add_argument("--torch_seed", default=94, type=int,
                        help="Numpy seed")

    args = parser.parse_args()
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    cResFreq_path = 'checkpoint/skipfreq_snr_big8/fr/epoch_60.pth'
    spcFreq_path = "checkpoint/freq_train_big8/fr/epoch_140.pth"
    deepfreq_path = "checkpoint/deepfreq_norm_snr40_big8/fr/deepfreq_epoch_120.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cResFreq_module, _, _, _, _ = util.load(cResFreq_path, 'skip', device)
    cResFreq_module.cpu()
    cResFreq_module.eval()
    spcFreq_module, _, _, _, _ = util.load(spcFreq_path, 'freq', device)
    spcFreq_module.cpu()
    spcFreq_module.eval()
    deepfreq_module, _, _, _, _ = util.load(deepfreq_path, 'fr', device)
    deepfreq_module.cpu()
    deepfreq_module.eval()
    x_len = 1024*4
    xgrid = np.linspace(-0.5, 0.5, x_len, endpoint=False)

    s, f, nfreq,r = gen_signal_accuracy(
        num_samples=args.n_test,
        signal_dim=args.signal_dimension,
        num_freq=args.max_freq,
        min_sep=args.minimum_separation,
        distance=args.distance,
        amplitude=args.amplitude,
        floor_amplitude=args.floor_amplitude,
        variable_num_freq=True)

    ff = 0.200001
    db=list(range(-15,18,3))

    eval_snrs = [np.float(x) for x in db]
    iter_num = 1000
    deepfreq_est = np.zeros([len(eval_snrs),iter_num])
    cResFreq_est = np.zeros([len(eval_snrs), iter_num])
    fft_est = np.zeros([len(eval_snrs),iter_num])
    spcFreq_est = np.zeros([len(eval_snrs),iter_num])
    music_est = np.zeros([len(eval_snrs),iter_num])
    omp_est = np.zeros([len(eval_snrs), iter_num])
    fc=np.zeros(len(eval_snrs))
    for k, snr in enumerate(eval_snrs):
        fc[k]=crlb(args.signal_dimension,snr)
        for iter in range(iter_num):
            print(k,iter)
            noisy_signals = noise.noise_torch(torch.tensor(s), snr, 'gaussian').cpu().numpy()
            signal_c = noisy_signals[:, 0] + 1j * noisy_signals[:, 1]

            with torch.no_grad():
                mv = np.max(np.sqrt(pow(noisy_signals[0][0], 2) + pow(noisy_signals[0][1], 2)))
                noisy_signals[0][0] = noisy_signals[0][0] / mv
                noisy_signals[0][1] = noisy_signals[0][1] / mv
                file = h5py.File('signal.h5', 'w')  # 创建一个h5文件，文件指针是f
                file['signal'] = noisy_signals[0]  # 将数据写入文件的主键data下面
                file.close()
                file = h5py.File('tgt_num.h5', 'w')  # 创建一个h5文件，文件指针是f
                file['tgt_num'] = nfreq[0]  # 将数据写入文件的主键data下面
                file.close()
                fft_sig = fft(signal_c, 256, 1)
                fft_sig = np.abs(fft_sig / np.max(np.abs(fft_sig), 1)[:, None])

                deepfreq_fr = deepfreq_module(torch.tensor(noisy_signals[0][None]))[0]
                spcFreq_fr = spcFreq_module(torch.tensor(fft_sig[0][None]).to(torch.float32))[0]
                cResFreq_fr = cResFreq_module(torch.tensor(noisy_signals[0][None]))[0]

            music_fr = fr.music(signal_c[0][None], xgrid, nfreq[0][None])[0]
            periodogram_fr = fr.periodogram(signal_c[0][None], xgrid)[0]
            omp_fr = eng.omp_func(nargout=1)
            omp_fr = omp_fr[0]

            deepfreq_fr = deepfreq_fr.cpu().data.numpy()
            f_hat = fr.find_freq_m(deepfreq_fr, nfreq[0], xgrid)
            deepfreq_est[k,iter]=f_hat[0,0]

            spcFreq_fr = spcFreq_fr.cpu().data.numpy()
            f_hat = fr.find_freq_m(spcFreq_fr, nfreq[0], xgrid)
            spcFreq_est[k,iter]=f_hat[0,0]


            cResFreq_fr = cResFreq_fr.cpu().data.numpy()
            f_hat = fr.find_freq_m(cResFreq_fr, nfreq[0], xgrid)
            cResFreq_est[k, iter] = f_hat[0,0]

            f_hat = fr.find_freq_m(music_fr, nfreq[0], xgrid)
            music_est[k, iter] =f_hat[0,0]


            f_hat = fr.find_freq_m(periodogram_fr, nfreq[0], xgrid)
            fft_est[k, iter] = f_hat[0,0]

            f_hat = fr.find_freq_m(omp_fr, nfreq[0], xgrid)
            omp_est[k, iter] = f_hat[0, 0]
            x=1


    # pickle.dump(fft_est, open('fft_acc.txt', 'wb'))
    # pickle.dump(music_est, open('music_acc.txt', 'wb'))
    # pickle.dump(deepfreq_est, open('deepfreq_acc.txt', 'wb'))
    # pickle.dump(spcFreq_est, open('spcFreq_acc.txt', 'wb'))
    # pickle.dump(cResFreq_est, open('cResFreq_acc.txt', 'wb'))
    #
    # fft_est = pickle.load(open('fft_acc.txt', 'rb'))
    # music_est = pickle.load(open('music_acc.txt', 'rb'))
    # deepfreq_est = pickle.load(open('deepfreq_acc.txt', 'rb'))
    # spcFreq_est = pickle.load(open('spcFreq_acc.txt', 'rb'))
    # cResFreq_est = pickle.load(open('cResFreq_acc.txt', 'rb'))

    acc_music=np.zeros([len(db),1])
    acc_fft=np.zeros([len(db),1])
    acc_deepfreq=np.zeros([len(db),1])
    acc_spcFreq=np.zeros([len(db),1])
    acc_cResFreq=np.zeros([len(db),1])
    acc_omp = np.zeros([len(db), 1])
    for i in range(len(db)):
        tmp=fft_est[i,:]-ff
        acc_fft[i]=np.sqrt(np.mean(pow(tmp,2)))
        tmp=music_est[i,:]-ff
        acc_music[i]=np.sqrt(np.mean(pow(tmp,2)))
        tmp=deepfreq_est[i,:]-ff
        acc_deepfreq[i]=np.sqrt(np.mean(pow(tmp,2)))
        tmp=spcFreq_est[i,:]-ff
        acc_spcFreq[i]=np.sqrt(np.mean(pow(tmp,2)))
        tmp=cResFreq_est[i,:]-ff
        acc_cResFreq[i]=np.sqrt(np.mean(pow(tmp,2)))
        tmp=omp_est[i,:]-ff
        acc_omp[i]=np.sqrt(np.mean(pow(tmp,2)))

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.98, wspace=None, hspace=None)
    plt.tick_params(labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax.set_xlabel('SNR / dB', size=20)
    ax.set_ylabel(r'RMSE / Hz', size=20)

    ax.semilogy(db,acc_fft[:,0],'--',c='m',marker='o',label='Periodogram',linewidth=4,markersize=10)
    ax.semilogy(db, acc_music[:, 0], '--', c='g', marker='o', label='MUSIC', linewidth=4, markersize=10)
    ax.semilogy(db, acc_omp[:, 0], '--', marker='o', label='OMP', linewidth=4, markersize=10)
    ax.semilogy(db,acc_spcFreq[:,0],'--',c='k',marker='o',label='spcFreq',linewidth=4,markersize=10)
    ax.semilogy(db, acc_deepfreq[:, 0], '--', c='b', marker='o', label='DeepFreq', linewidth=4, markersize=10)
    ax.semilogy(db, acc_cResFreq[:,0], '--', c='r', marker='o', label='cResFreq', linewidth=4,markersize=10)
    ax.semilogy(db, fc, '--', c='#EDB120', marker='o', label='CRLB', linewidth=5, markersize=10)
    # ax.set_ylim(2.8e-3,1e-1)
    plt.grid(linestyle='-.')

    plt.legend(frameon=True, prop={'size': 16})
    x=1
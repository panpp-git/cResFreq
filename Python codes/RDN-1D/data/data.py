import numpy as np
import torch


def frequency_generator(f, nf, min_sep, dist_distribution):
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'jittered':
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep)


def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep, scale=0.05):
    """
    Distance between two frequencies follows a normal distribution
    """
    f[0] = np.random.uniform() - 0.5
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale)
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (15 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return 5*np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)



def gen_signal(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq
    for n in range(num_samples):
        if n%100==0:
            print(n)
        # r[0,0]=0.12
        # r[0,1]=3
        # nfreq[0]=2
        # r[0,0:2]=np.array([0.15,15])

        # if nfreq[n]>=2 and np.random.rand()>=0:
        #     nfreq[n]=2
        #     idx=np.argmax(r[n,0:nfreq[n]])
        #     r[n,idx]=10 ** (41 / 20) * np.min(r[n, :nfreq[n]])


        frequency_generator(f[n], nfreq[n], d_sep, distance)
        # f[0,0:2]=np.array([0,0.2])
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / (np.sqrt(np.mean(np.power(s[n], 2)))+1e-13)
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r

def gen_signal_accuracy(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        nfreq[n]=1
        r[n,0]=1

        frequency_generator(f[n], nfreq[n], d_sep, distance)
        f[n,0]=0.200001
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r

def gen_signal_resolution(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,dw=None):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        nfreq[n]=2
        r[n,0:2]=np.array([1,1])

        frequency_generator(f[n], nfreq[n], d_sep, distance)
        f[n,0:2]=np.array([0,dw/64])
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r

def gen_signal_weak(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        nfreq[n]=2

        r[n, 1] = np.abs(np.random.randn()) + 0.1
        r[n, 0] = 10 ** (20/ 20) * r[n, 1]



        frequency_generator(f[n], nfreq[n], d_sep, distance)

        #signal for testing the matched filter module
        if n==0:
            nfreq[0]=1
            r[0,0]=1
            f[0,0:10]='inf'
            f[0,0]=0.0

        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r


def gen_signal_weak2(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
                    floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    amp_diff=np.array(range(8,35,2))
    for n in range(num_samples):
        nfreq[n] = 2

        r[n, 1] = np.abs(np.random.randn()) + 0.1
        r[n, 0] = 10 ** (amp_diff[n//1000] / 20) * r[n, 1]

        frequency_generator(f[n], nfreq[n], d_sep, distance)

        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j * theta[n, i] + 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq, r

def compute_snr(clean, noisy):
    return np.linalg.norm(clean, axis=1) ** 2 / np.linalg.norm(clean - noisy, axis=1) ** 2


def compute_snr_torch(clean, noisy):
    return (torch.sum(clean.view(clean.size(0), -1) ** 2, dim=1) / torch.sum(
        ((clean - noisy).view(clean.size(0), -1)) ** 2, dim=1)).mean()

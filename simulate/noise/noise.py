# filter some data using SOS so as to preserve the dynamic range

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


def generate_noise():
    fs = 1024

    bg_raw = np.random.randn(1000)

    # This is a bandpass
    f1    = 15
    f2    = 100
    sosBP1 = sig.butter(8, [f1/(fs/2), f2/(fs/2)], btype='bandpass', output='sos')

    # invertable band pass
    fz = [4, 400]
    fp = [15, 20, 150]

    _,z0,_ = sig.butter(10, fz[0]/(fs/2), btype='lowpass', output='zpk')
    _,p0,_ = sig.butter( 8, fp[0]/(fs/2), btype='lowpass', output='zpk')
    _,p1,_ = sig.butter( 1, fp[1]/(fs/2), btype='lowpass', output='zpk')
    _,p2,_ = sig.butter( 3, fp[1]/(fs/2), btype='lowpass', output='zpk')
    _,z1,_ = sig.butter( 2, fz[1]/(fs/2), btype='lowpass', output='zpk')

    zs    = np.concatenate((z0, z1))
    ps    = np.concatenate((p0, p1))
    BP    = sig.zpk2sos(zs, ps, 1, pairing='keep_odd')
    sosBP = np.concatenate([sosBP1, BP], axis = 0)

    # filter the data using many second-order-sections
    bg    = sig.sosfilt(sosBP,  bg_raw)
    return bg


if __name__ == '__main__':
    bg = generate_noise()
    plt.plot(bg)
    plt.show()



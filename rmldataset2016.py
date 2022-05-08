import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

def APPROCESS(iqsignal):
    XX=(iqsignal[:,0,:]**2+iqsignal[:,1,:]**2)**0.5
    YY=np.arctan(iqsignal[:,1,:]/iqsignal[:,0,:])
    XXX=preprocessing.normalize(XX, norm='l2')
    for i in range(iqsignal.shape[0]):
        YY[i,:]=YY[i,:]/np.max(np.abs(YY[i,:]))
    APsignal=np.dstack((XXX,YY))
    return APsignal

def load_data(filename = "data/RML2016.10a_dict.pkl",train_rate = 0.5):

    Xd = pickle.load(open(filename,'rb'),encoding='iso-8859-1')


    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]

    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            xx = Xd[(mod, snr)]
            X.append(Xd[(mod, snr)])  # ndarray(1000,2,128)
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X)  # (220000,128,2)  mods * snr * 1000,total 220000 samples


    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * train_rate)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = np.vstack((X[train_idx],X[train_idx],X[train_idx],X[train_idx]))
    # X_train = X[train_idx]
    X_train[110000:219999,0,:]=-X_train[110000:219999,0,:]
    X_train[220000:329999,1, :] = -X_train[220000:329999,1, :]
    X_train[330000:439999,:,:]=-X_train[330000:439999,:,:]
    X_test =  X[test_idx]
    # X_train=APPROCESS(X_train)
    # X_test = APPROCESS(X_test)
    # print(X_train[0,:,:])
    # print("--------------------------")
    # print(X_train[220000, :, :])



    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1
    # yy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    YY_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_train = np.vstack((YY_train, YY_train,YY_train,YY_train))
    # print(Y_train[0,:])
    # print(Y_train[110000, :])
    # print(Y_train[330000, :])
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    X_train=X_train.swapaxes(2,1)
    X_test=X_test.swapaxes(2,1)
    return (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx)
# import scipy.signal as signal
# def stft(x, **params):
#     '''
#     :param x: 输入信号
#     :param params: {fs:采样频率；
#                     window:窗。默认为汉明窗；
#                     nperseg： 每个段的长度，默认为256，
#                     noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
#                     nfft：fft长度，
#                     detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
#                     return_onesided：默认为True，返回单边谱。
#                     boundary：默认在时间序列两端添加0
#                     padded：是否对时间序列进行填充0（当长度不够的时候），
#                     axis：可以不必关心这个参数}
#     :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
#     '''
#     f, t, zxx = signal.stft(x, **params)
#     return f, t, zxx
#
# def stft_specgram(x, picname=None, **params):    #picname是给图像的名字，为了保存图像
#     f, t, zxx = stft(x, nperseg=128)
#     plt.pcolormesh(t, f, np.abs(zxx))
#     plt.colorbar()
#     plt.title('STFT Magnitude')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.tight_layout()
#     if picname is not None:
#         plt.savefig('..\\picture\\' + str(picname) + '.jpg')       #保存图像
#     plt.show()      #清除画布
#     return t, f, zxx






if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx)= load_data()
    # z=np.array([[[1,2,3,-1,-3],[1,2,3,-2,-1]],[[1,2,2,-1,-3],[1,2,1,-2,-1]],[[1,3,3,-1,-2],[1,11,3,-1,-1]]])
    # Z_amp = np.abs(z[:, 0, :] + 1j * z[:, 1, :])
    # Z_ang = np.arctan2(z[:, 1, :], z[:, 0, :])
    #
    # ZZ=np.dstack((Z_amp, Z_ang))
    # for i in range(3):
    #      z[i,1,:]=z[i,1,:]/np.max(np.abs(z[i,1,:]))


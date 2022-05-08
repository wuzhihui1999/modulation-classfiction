﻿# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)

import numpy as np
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from matplotlib import pyplot as plt
import pickle, random, sys,h5py
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler

from keras.regularizers import *

from keras.models import model_from_json
from keras.utils.vis_utils import plot_model

import mltools,rmldataset2016
#import rmlmodels.CNN2Model as cnn2
#import rmlmodels.ResNetLikeModel as resnet
#import rmlmodels.VGGLikeModel as vggnet
import CLDNNLikeModel as cldnn
# from tftb.processing import PseudoWignerVilleDistribution

#set Keras data format as channels_first
K.set_image_data_format('channels_last')
print(K.image_data_format())

(mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx) = \
    rmldataset2016.load_data(filename ="data/RML2016.10a_dict.pkl", train_rate = 0.5)


in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
print(classes)



# Set up some params
nb_epoch = 120     # number of epochs to train on
batch_size = 1024  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
#model = cnn2.CNN2Model(None, input_shape=in_shp,classe=len(classe))

# model.compile(loss='catagorical_crossentropy',metrics=['accuracy'],optimizer='adam')
# rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model = cldnn.CLDNNLikeModel(None,input_shape=[128,2])
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')
plot_model(model, to_file='model_CLDNN.png',show_shapes=True) # print model
model.summary()

def scheduler(epoch):
    print("epoch({}) lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

filepath = 'weights/CLDNN_dr0.5.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks = [reduce_lr,
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patince=3,min_lr=0.000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')

                ]
                    )

# we re-load the best weights once training is finished
# model.load_weights(filepath)
mltools.show_history(history)

#Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

def plot_tSNE(model,filename="data/RML2016.10a_dict.pkl"):
    from keras.models import Model
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = \
    #     rmldataset2016.load_data(filename, train_rate=0.50)

    #设计中间层输出的模型
    dense2_model = Model(inputs=model.input,outputs=model.get_layer('fc1').output)

    #提取snr下的数据进行测试
    for snr in [s for s in snrs if s > 14]:
        test_SNRs = [lbl[x][1] for x in test_idx]       #lbl: list(mod,snr)
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        #计算中间层输出
        dense2_output = dense2_model.predict(test_X_i,batch_size=32)
        Y_true = np.argmax(test_Y_i,axis=1)

        #PCA降维到50以内
        pca = PCA(n_components=50)
        dense2_output_pca = pca.fit_transform(dense2_output)

        #t-SNE降为2
        tsne = TSNE(n_components=2,perplexity=5)
        Y_sne = tsne.fit_transform(dense2_output_pca)



        # 散点图
        # plt.scatter(Y_sne[:,0],Y_sne[:,1],s=5.,color=plt.cm.Set1(Y_true / 11.),label=classes)

        # 标签图
        data = Y_sne
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        for i in range(Y_sne.shape[0]):
            plt.text(data[i,0],data[i,1], str(Y_true[i]))
            plt.title('t-SNR at snr:{}'.format(snr))

        # plt.legend()  # 显示图示
        # fig.show()

def predict(model,filename="data/RML2016.10a_dict.pkl"):
    # (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = \
    #     rmldataset2016.load_data(filename, train_rate=0.7)
    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm,_,_ = mltools.calculate_confusion_matrix(Y_test,test_Y_hat,classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes,save_filename='figure/total_confusion')

    # Plot confusion matrix
    acc = {}
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )
    i = 0
    for snr in snrs:

        # extract classes @ SNR
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(test_Y_i,test_Y_i_hat,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)

        mltools.plot_confusion_matrix(confnorm_i, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)(ACC=%2f)" % (snr,100.0*acc[snr]),save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr,100.0*acc[snr]))

        acc_mod_snr[:,i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i,axis=1),3)
        i = i +1

    #plot acc of each mod in one picture
    dis_num=11
    for g in range(int(np.ceil(acc_mod_snr.shape[0]/dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g*dis_num
        end_index = np.min([(g+1)*dis_num,acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index,end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            # 设置数字标签
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        plt.close()
    #save acc for mod per SNR
    fd = open('predictresult/acc_for_mod_on_cldnn.dat', 'wb')
    pickle.dump(('128','cldnn', acc_mod_snr), fd)
    fd.close()

    # Save results to a pickle file for plotting later
    print(acc)
    fd = open('predictresult/CLDNN_dr0.5.dat','wb')
    pickle.dump( ("1D", 0.5, acc) , fd )

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CLDNN Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')


if __name__ == '__main__':
    plot_tSNE(model)
    predict(model, filename="data/RML2016.10a_dict.pkl")

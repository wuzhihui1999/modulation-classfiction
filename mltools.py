import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
#from matplotlib import pyplot as plt
import numpy as np
import pickle

# Show loss curves
def show_history(history):
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig('figure/total_loss.png')
    plt.close()
 

    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history['accuracy'], label='train_acc')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_acc')
    plt.legend()    
    plt.savefig('figure/total_acc.png')
    plt.close()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[],save_filename=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_filename is not None:
        plt.savefig(save_filename)
    plt.close()

def calculate_confusion_matrix(Y,Y_hat,classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes,n_classes])
    confnorm = np.zeros([n_classes,n_classes])

    for k in range(0,Y.shape[0]):
        i = list(Y[k,:]).index(1)
        j = int(np.argmax(Y_hat[k,:]))
        conf[i,j] = conf[i,j] + 1

    for i in range(0,n_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    # print(confnorm)

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm,right,wrong

def calculate_accuracy_each_snr(Y,Y_hat,Z,classes=None):
    Z_array = Z[:,0]
    snrs = sorted(list(set(Z_array)))
    # snrs = np.arange(-20,32,2)
    acc = np.zeros(len(snrs))

    Y_index = np.argmax(Y,axis=1)
    Y_index_hat = np.argmax(Y_hat,axis=1)

    i = 0
    for snr in snrs:
        Y_snr = Y_index[np.where(Z_array == snr)]
        Y_hat_snr = Y_index_hat[np.where(Z_array == snr)]

        acc[i] = np.sum(Y_snr==Y_hat_snr)/Y_snr.shape[0]
        i = i +1

    plt.figure(figsize=(8, 6))
    plt.plot(snrs,acc, label='test_acc')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2018.01")
    plt.legend()
    plt.grid()
    plt.show()

'''
?????????snr??????????????????????????????snr????????????????????????acc,????????????
'''
def calculate_acc_at1snr_from_cm(cm):
    return np.round(np.diag(cm)/np.sum(cm,axis=1),3)

def calculate_acc_cm_each_snr(Y,Y_hat,snrs,cm,classes=None,save_figure=True,min_snr = 0):
    # Z_array = Z[:,0]
    # snrs = sorted(list(set(Z_array)))
    acc = np.zeros(len(snrs))

    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )  #mods*snrs,24*26
    i = 0
    for snr in snrs:
        Y_snr = Y[np.where(Z_array == snr)]
        Y_hat_snr = Y_hat[np.where(Z_array == snr)]

        # plot confusion for each snr
        cm,right,wrong = calculate_confusion_matrix(Y_snr,Y_hat_snr,classes)
        print(min_snr)
        if snr >= min_snr:
            plot_confusion_matrix(cm, title='Confusion matrix at {}db'.format(snr), cmap=plt.get_cmap("Blues"), labels=classes,save_filename = 'figure/cm_snr{}.png'.format(snr))

        # cal acc on each snr
        acc[i] = round(1.0*right/(right+wrong),3)
        print('Accuracy at %ddb:%.2f%s / (%d + %d)' % (snr,100*acc[i],'%',right, wrong))

        acc_mod_snr[:,i] = calculate_acc_at1snr_from_cm(cm)

        i = i +1

    '''
    acc??????snr???????????????
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(snrs,acc, label='test_acc')
    # ??????????????????
    for x, y in zip(snrs,acc):
        plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on All Test Data")
    plt.legend()
    plt.grid()
    plt.savefig('figure/acc_overall.png')
    plt.show()

    fd = open('acc_overall_128k_on_512k_wts.dat', 'wb')
    pickle.dump(('128k','512k', acc), fd)
    fd.close()

    '''
    acc??????snr???????????????,??????mod????????????
    '''
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
            # ??????????????????
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        if save_figure:
            plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        plt.close()
    
    fd = open('predictresult/acc_for_mod_on_vgg.dat', 'wb')
    pickle.dump(('128','vgg', acc_mod_snr), fd)
    fd.close()
    # print(acc_mod_snr)

def main():
    import rmldataset2016
    import numpy as np
    (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx) = \
        rmldataset2016.load_data(filename ="RML2016.10a_dict.pkl", train_rate = 0.2)

    one_sample = X_test[0]
    print(np.shape(one_sample))
    print(one_sample[0:2])
    print(np.max(one_sample,axis=1))
    one_sample = np.power(one_sample,2)
    one_sample = np.sqrt(one_sample[0,:]+one_sample[1,:])

    plt.figure()
    plt.title('Training Samples')
    one_sample_t = np.arange(128)
    plt.plot(one_sample_t,one_sample)
    # plt.scatter()
    plt.grid()
    plt.show()

    sum_sample = np.sum(one_sample)
    print(sum_sample)
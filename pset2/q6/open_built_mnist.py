
import pickle 
from os.path import dirname, join as pjoin
import scipy.io as sio 
from sklearn.preprocessing import Normalizer
import numpy as np

def save_to_pickle(obj,filename):
    with open(filename,'wb') as outp:
        pickle.dump(obj,outp,pickle.HIGHEST_PROTOCOL)
    
def load_pickle(picklename):
    with open(picklename,'rb') as inp:
        obj = pickle.load(inp)
        return obj

def save_mnist(filename):

    mnist = sio.loadmat(filename, struct_as_record=True,
    verify_compressed_data_integrity=True,matlab_compatible=True)
    save_to_pickle(mnist,'mnistpickle.pkl')


#normalizes images to havec unit Euclidean norm.
#returns mnist with normalized X
def normalize_mnist(mnist):
    X_train = np.array(mnist['trainX'],dtype = float)
    X_test = np.array(mnist['testX'], dtype =float)

    X_train_l2 = np.linalg.norm(X_train,axis=1)
    X_test_l2 = np.linalg.norm(X_test,axis=1)

    
    for i in range(len(X_train)):
        X_train[i] = X_train[i]/X_train_l2[i]
    #X_train /=  X_train_l2
    for i in range(len(X_test)):
        X_test[i] = X_test[i]/X_test_l2[i]

    mnist['trainX'] = X_train
    mnist['testX'] = X_test 


    

def combine_train_and_test(mnist):
    X = np.append(mnist['trainX'],mnist['testX'],axis=0)
    Y = np.append(mnist['trainY'],mnist['testY'],axis=0)
    return X, Y 

def main():
    #save_mnist('./data/mnist.mat')
    #mnist = load_pickle('data/mnist_normalized.pkl')
    
    #normalize_mnist(mnist)
    
    #save_to_pickle(mnist, 'mnist_normalized.pkl')

    #print(mnist['trainX'][0])



if __name__ == "__main__":
    main()

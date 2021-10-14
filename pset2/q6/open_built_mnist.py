
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

def l2_norm(vals):

    new_vals = np.zeros(len(vals))

    for i in range(len(vals)):
      
        new_vals[i] = np.linalg.norm(vals[i])
    
    return new_vals

def normalize_mnist(mnist):

    for i in range(len(mnist['testX'])):
        print(np.linalg.norm(mnist['testX'][i]))
        print(np.linalg.norm(mnist['testY'][i]))

    pass 

def combine_train_and_test(mnist):
    X = np.append(mnist['trainX'],mnist['testX'],axis=0)
    Y = np.append(mnist['trainY'],mnist['testY'],axis=0)
    return X, Y 

def main():
    #save_mnist('./data/mnist.mat')
    mnist = load_pickle('data/mnistpickle.pkl')
    '''
    print(obj.keys())
    print('header: ' , obj['__header__'])
    print('version: ' , obj['__version__'])
    print('globals: ', obj['__globals__'])
    print('example x and y: \n\n X: \n ', obj['testX'][0], ' \n\n Y:\n ', obj['testY'][0])
    
    '''
    X, Y = combine_train_and_test(mnist)

    X = l2_norm(X)


if __name__ == "__main__":
    main()

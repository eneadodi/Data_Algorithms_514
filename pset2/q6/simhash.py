
import pickle 
from os.path import dirname, join as pjoin
import scipy.io as sio 
from sklearn.preprocessing import Normalizer
import numpy as np
import collections 
import time
def load_pickle(picklename):
    with open(picklename,'rb') as inp:
        obj = pickle.load(inp)
        return obj

def make_random_vector(shape):
    return np.random.normal(size=shape)

class SimHashTable():

    def __init__(self, min_cos_sim, signature_length, input_shape):
        self.min_cos_sim = min_cos_sim
        #self.t = table_repetitions
        self.r = signature_length
        self.input_shape = input_shape
    
        self.r_vectors = np.array([self.make_random_vector() for x in range(self.r)])
        self.hash_table = collections.defaultdict(list) #np.array([collections.defaultdict(list) for x in range(self.t)])


    def make_random_vector(self):
        return np.random.normal(size=self.input_shape)

    def simhash(self,v,all = False,specific_vector = 1):
        if all:
            l = []
            for i in range(self.r):
                l.append(np.sign(np.dot(v,self.r_vectors[i])))
            return l
        else:
            return np.sign(np.sign(np.dot(v,self.r_vectors[specific_vector])))
        

    def generate_bit_codes(self,values):
        def convert_to_bytes(x):
            if x == -1:
                return '0'
            else:
                return '1'

        for v in values:
            code = self.simhash(v,all=True)

            code = ''.join(map(convert_to_bytes,code))
            self.hash_table[hash(code)].append(v)
            
class SimHashFull():

    def __init__(self,min_cos_sim, signature_length,table_repetitions, input_shape):
        self.min_cos_sim = min_cos_sim
        self.t = table_repetitions
        self.r = signature_length
        self.input_shape = input_shape
        self.tables = [SimHashTable(self.min_cos_sim,self.r,self.input_shape) for x in range(self.t)]

    def fill_tables(self,values):
        for i in range(self.t):
            self.tables[i].generate_bit_codes(values)

def checking_most(simh):
    maxcount = max(len(v) for v in simh.hash_table.values())
    most = [k for k,v in simh.hash_table.items() if len(v) == maxcount]
    print(most[0])
    print(len(simh.hash_table[most[0]]))

def main():
    mnist = load_pickle('data/mnist_normalized.pkl')

    #simh = SimHashTable(0.95,10,mnist['trainX'][0].shape)
    #simh.generate_bit_codes(mnist['trainX'])
    
    # checking_most(simh) THIS WORKS!

    start = time.time()
    simhash_full = SimHashFull(0.95,10, 5, mnist['trainX'][0].shape)
    simhash_full.fill_tables(mnist['trainX'])
    end = time.time()

    print('total time: ' ,(end-start), ' seconds :)')

if __name__ == "__main__":
    main()
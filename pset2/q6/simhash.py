
import pickle 
from os.path import dirname, join as pjoin
import scipy.io as sio 

import numpy as np
import collections 
import time
import random
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

def save_to_pickle(obj,filename):
    with open(filename,'wb') as outp:
        pickle.dump(obj,outp,pickle.HIGHEST_PROTOCOL)

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
        self.hash_table = collections.defaultdict(list) 


    def make_random_vector(self):
        return np.random.normal(size=self.input_shape)

    def simhash(self,v,all = False,specific_vector = 1):
        if all:
            l = []
            for i in range(self.r):
                l.append(np.sign(np.dot(v,self.r_vectors[i])))
            return l
        else:
            return np.sign(np.dot(v,self.r_vectors[specific_vector]))
        

    def generate_bit_codes(self,values,count_collisions=False, x_col=None):
        def convert_to_bytes(x):
            if x == -1:
                return '0'
            else:
                return '1'    

        reference_codes = set()
        for i in range(len(values)):

            v = values[i]
            
            code = self.simhash(v,all=True)

            code = ''.join(map(convert_to_bytes,code))
            #print(code)
            hash_code = hash(code)
            reference_codes.add(hash_code)
            self.hash_table[hash_code].append((i,code)) #i is the index in mnist, code is the code created
        
        if count_collisions:
            return reference_codes 
            
class SimHashFull():

    def __init__(self,min_cos_sim, signature_length,table_repetitions, input_shape):
        self.min_cos_sim = min_cos_sim
        self.t = table_repetitions
        self.r = signature_length
        self.input_shape = input_shape
        self.tables = [SimHashTable(self.min_cos_sim,self.r,self.input_shape) for x in range(self.t)]

    def fill_tables(self,values,count_collisions = False, collision_x=None):
        
        reference_hashes = []

        for i in range(self.t):
            if count_collisions:
                reference_hashes.append(list(self.tables[i].generate_bit_codes(values,count_collisions,collision_x)))
            else:
                self.tables[i].generate_bit_codes(values)
        
        if count_collisions:
            return reference_hashes

    def add_table(self,values):
        self.t = self.t + 1
        self.tables.append(SimHashTable(self.min_cos_sim,self.r,self.input_shape))
        self.tables[t-1].generate_bit_codes(values)
    

def checking_most(simh):
    maxcount = max(len(v) for v in simh.hash_table.values())
    most = [k for k,v in simh.hash_table.items() if len(v) == maxcount]
    return most[0]


def finding_occurences_q4(simh,mnist,length=6):
    for key,value in simh.hash_table.items():
        if len(value) >= 6 and len(value) <= 7:
            return key

def cosine_similarity(x,y):
    inner = np.inner(x,y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    similarity = inner / (np.dot(norm_x,norm_y))
    return similarity

def q1():
    desired_result = 0.02
    r_t_pairs = []
    for r in range(1,31):    
        tb = math.log(desired_result)/math.log(1-(0.95**r))
        #print(math.ceil(tb))

        t = 1
        while True:
                
            result = (1-(0.95**r))**t 
            if result <= desired_result:
                print('r = ', r)
                print('t = ', t)
                print('tb = ' , tb)
                print('\n------\n')
                r_t_pairs.append((r,t))
                break
            else:
                t+=1
    return r_t_pairs



def make_plot(images_index):
    #load original mnist:
    og_mnist = load_pickle('data/mnistpickle.pkl')
    images = []
    for i in images_index: 
        images.append(np.reshape(og_mnist['trainX'][i],(28,28)))
    
    fig = plt.figure(figsize=(15,12))

    rows = math.ceil(len(images)/2)
    cols = math.floor(len(images)/2)

    for i in range(len(images)):
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title('index = ' + str(images_index[i]))

    plt.savefig('q6.png')

        
def q4(mnist):
    X = mnist['testX']
    Y = mnist['testY']
    length = len(X[0])
    
    simh = SimHashTable(0.95,35,length)
    simh.generate_bit_codes(X)

    occurences_hash_value = finding_occurences_q4(simh,mnist)

    bucket = simh.hash_table[occurences_hash_value]

    images_index = []
    for i,j in bucket:
        images_index.append(i)
    return images_index


def q3(pairs,mnist):

    X = mnist['testX']
    Y = mnist['testY']
    length = len(X[0])

    total_counter = []
    z = 0
    for r,t in pairs:
        
        simhash_full = SimHashFull(0.95,r,t,length)

        #this is a 2D list where the j are the values in table i
        reference_hashes = simhash_full.fill_tables(X,True)

        collision_counter = 0
        collisions_in_bucket = collections.defaultdict(int)

        for i in range(len(reference_hashes)):
            for j in range(len(reference_hashes[i])):
                total_collisions_for_current_hash_and_table = (len(simhash_full.tables[i].hash_table[reference_hashes[i][j]]) - 1)
                collisions_in_bucket[str(reference_hashes[i][j])] += total_collisions_for_current_hash_and_table
        
        length_collisions = 0
        sum_collisions = 0
        for key,value in collisions_in_bucket.items():
            length_collisions += 1
            sum_collisions += value 
            
        
        total_counter.append(sum_collisions/length_collisions)

        #
    
    #finally make a dictionary for r values and total_countfer
    return_dict = {}
    i = 0
    for r,t in pairs :
        return_dict[r] = total_counter[i]
        i+=1

    
    return return_dict 


def main():
    #mnist = load_pickle('data/mnist_normalized.pkl')

    #r_t_pairs = q1()

    #expected_collisions = q3(r_t_pairs,mnist)
    #save_to_pickle(expected_collisions,'q3results.pkl')
   
    #indeces = q4(mnist)
    #make_plot(indeces)

    q3_vals = load_pickle('q3results.pkl')
    print(q3_vals)
    x = []
    y = []
    for key,value in q3_vals.items():
        x.append(key)
        y.append(value)
    
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x,y)
    plt.savefig('q3.png')
    plt.title('r vs expected collisions')
if __name__ == "__main__":
    main()
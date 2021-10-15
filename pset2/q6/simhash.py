
import pickle 
from os.path import dirname, join as pjoin
import scipy.io as sio 
from sklearn.preprocessing import Normalizer
import numpy as np


class SimHash():

    def __init__(self, min_cos_sim, table_repetitions, signature_length):
        self.min_cos_sim =min_cos_sim
        self.t = table_repetitions
        self.r = signature_length
        

def main():
    print('hello')

if __name__ == "__main__":
    main()
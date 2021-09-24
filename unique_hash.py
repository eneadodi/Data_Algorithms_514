import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numba as nb
import random
import collections 

def graph(c,name):
    plt.bar(*zip(*c.items()))
    plt.savefig(name)
    plt.clf()



def h(t,r=0.07):
    x_new = random.random()
    #y_new = random.random()
    for i in range(t):
        x_new = r*x_new*(1-x_new)# - s*(y_new*x_new)
        #y_new = r*y_new*(1-y_new) #-  s*(x_new*y_new)
    return int(np.floor(t/(x_new) % 100))

def main():
    data = h(1)
    h_1 = []
    h_2 = []
    h_1minus2 = []

    for i in range(10000):
        h_1.append(h(1))
        h_2.append(h(2))
        h_1minus2.append((h_1[i]-h_2[i]))
    
    counter_h1 = collections.Counter(h_1)
    counter_h2 = collections.Counter(h_2)
    counter_h3 = collections.Counter(h_1minus2)

    graph(counter_h1,"h1.png")
    graph(counter_h2,"h2.png")
    graph(counter_h3,"h1-h2.png")

if __name__ == "__main__":
    main()
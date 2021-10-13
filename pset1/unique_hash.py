import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

import random
import collections 

def graph(counter,name):

    plt.bar(*zip(*counter.items()))
    
    sum_val = sum(counter.values())
    mean = sum_val / len(counter)

    plt.title('mean: ' + str(mean))
    plt.savefig(name)
    plt.clf()

def q2():
    h_vals = []

    t_vals = np.arange(1,20)

    for j in t_vals:
        h_vals = []
        for i in range(10000):
            h_vals.append(h(j,r=random.random() + 0.02,x=random.random()))
        
        counter_h1 = collections.Counter(h_vals)

        graph(counter_h1,"h"+str(j)+".png")


def q3():
    h_vals_1 = [] 
    h_vals_2 = []

    t_vals_1 = [1,2,3,5,6,12,23,63]
    t_vals_2 = [2,3,13,25,26,27,44,4]

    collisions_dict = {}

    for j in range(len(t_vals_1)):
        collisions_counter = 0
        for i in range(10000):
            x = random.random()
            r = random.random() + 0.02
            h_vals_1.append(h(t_vals_1[j],r=r,x=x))
            h_vals_2.append(h(t_vals_2[j],r=r,x=x))

            if h_vals_1[i] == h_vals_2[i]:
                collisions_counter+=1

        collisions_dict["x="+str(t_vals_1[j])+","+str(t_vals_2[j])] = collisions_counter

    graph(collisions_dict,"collisions_for_pairs.png")

def q4():
    h_vals = [] 

    t_vals_1 = [1,2,3,5,6,12,23,63]
    t_vals_2 = [2,3,13,25,26,27,44,4]



    for j in range(len(t_vals_1)):
        h_vals = []

        for i in range(10000):
            x = random.random()
            r = random.random() + 0.02
            h_vals.append(h(t_vals_1[j],r=r,x=x)-h(t_vals_2[j],r=r,x=x))

        counter_h = collections.Counter(h_vals)

        graph(counter_h,"h("+str(t_vals_1[j])+")-h("+str(t_vals_2[j])+").png")

def h(t,r=0.02,x=0.1):
    x_new = x
    #y_new = random.random()
    for i in range(t+10):
        x_new = r*x_new*(1-x_new)# - s*(y_new*x_new)
        #y_new = r*y_new*(1-y_new) #-  s*(x_new*y_new)
    return int(t/x_new) % 100


def main():
    q2()
    q3()
    q4()
if __name__ == "__main__":
    main()
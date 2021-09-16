#! /usr/bin/env python
# -*- coding: utf-8 -*-

from py_interface import *
from ctypes import *
import sys
import ns3_util
import time
import numpy as np
from tensorflow import keras
import random

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('throughput', c_double)
    ]


class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('next_rate', c_ubyte)
    ]


f = open('radius_5.txt', 'w')

total_episodes = 100
simulationTime = 10
m_nIntervals  = 100

nVhtStations=1
radius=5

mempool_key = 12344 # memory pool key, arbitrary integer large than 1000
mem_size = 4096 # memory pool size in bytes
exp = Experiment(mempool_key, mem_size, 'adaptiverate', '../../')
memblock_key = 2333 # memory block key in the memory pool, arbitrary integer, and need to keep the same in the ns-3 script

with open("trafficFile", "w") as text_file:
    text_file.write("1 %d BE UDP DL 1000 CBR 200" % (nVhtStations))
    
# make Q-network
model = keras.Sequential()
model.add(keras.layers.Dense(3, input_shape=(9,), activation='tanh'))
adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
              loss='mean_squared_error',
              metrics=['accuracy'])

total_thp = list()
last_mcs = list()

for episodes in range(total_episodes):
    print("episode: ", episodes+1)
    epsilon = 1 - episodes/(total_episodes/2)
    mcs = 0
    state = np.zeros(9)
    state[mcs] = 1
    state = np.reshape(state, [1, 9])

    # epsilon greedy
    if np.random.rand(1) < epsilon:
        action = np.random.randint(0, 3)
#        print("   random", action)
    else:
        res = model.predict(state)[0]
        action = np.argmax(res)
#        print("   not random", action)
#        print(res)

    if(action==0 and mcs!=8 ):
        mcs += 1
    elif(action==2 and mcs!=0):
        mcs -= 1

    #next_state
    next_state =  np.zeros(9)
    next_state[mcs] = 1
    next_state = np.reshape(next_state, [1, 9])

    throughputsum = 0
    last_throughput = 0
    exp.reset()
    rl = Ns3AIRL(memblock_key, Env, Act)
    ns3Settings = {'nVhtStations': nVhtStations, 'radius': radius, 'Rate': mcs, 'simulationTime': simulationTime, 'm_nIntervals': m_nIntervals}
    pro = exp.run(setting=ns3Settings, show_output=False)
#        print("  run wifi-ofmda", ns3Settings)
    while not rl.isFinish():
        with rl as data:
            if data == None:
                break
            print("   MCS: ", mcs)
            print("   Throughput: ",data.env.throughput)
            throughputsum += float(data.env.throughput)

            #Train
            if(data.env.throughput == 0):
                reward = -100
            else:
                reward = float(data.env.throughput) - last_throughput
            target = (reward + 0 * np.amax(model.predict(next_state)[0]))
            target_f = model.predict(state)
            target_f[0][action] = target
#            print(target_f)
            model.fit(state, target_f, epochs=1, verbose=0)

            state = next_state

           # epsilon greedy
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, 3)
                print("   random", action)
            else:
                res = model.predict(state)[0]
                action = np.argmax(res)
                print("   not random", action)
                print(res)

            if(action==0 and mcs!=8 ):
                mcs += 1
            elif(action==2 and mcs!=0):
                mcs -= 1
            data.act.next_rate = c_ubyte(mcs)

            #next state
            next_state =  np.zeros(9)
            next_state[mcs] = 1
            next_state = np.reshape(next_state, [1, 9])

            last_throughput = float(data.env.throughput)
    pro.wait()

    total_thp.append(throughputsum)
    last_mcs.append(mcs)
    f.write(str(throughputsum))
    f.write(', ')
del exp

print(total_thp)
print(last_mcs)

f.close()
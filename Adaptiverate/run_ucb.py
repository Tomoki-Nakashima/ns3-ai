#! /usr/bin/env python
# -*- coding: utf-8 -*-

from py_interface import *
from ctypes import *
import sys
import ns3_util
import time
import numpy as np
import matplotlib.pyplot as plt


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
    
def next_action(avg_reward,step,selected_time):
    next=np.argmax(avg_reward+np.sqrt(2*np.log(np.repeat(step,9))/(selected_time+np.repeat(1,9))))
    return next

f = open('ucb.txt', 'a')
    
nVhtStations=1
radius=1
Rate=0
mcs=str(Rate)
simTime=10
m_nIntervals=100
bandwidth=20

#total_reward_vec= np.zeros(0)
#total_tput_vec= np.zeros(0)

numMCS=9
#mcs_reward=np.zeros([numMCS,m_nIntervals*total_episode])
mcs_reward=[[ 0 for i in range(1)] for j in range(numMCS)]
selected_time=np.zeros(numMCS)
snr=64
avg_reward=np.zeros(numMCS)

mempool_key = 12344 # memory pool key, arbitrary integer large than 1000
mem_size = 4096 # memory pool size in bytes
exp = Experiment(mempool_key, mem_size, 'adaptiverate', '../../')
memblock_key = 2333 # memory block key in the memory pool, arbitrary integer, and need to keep the same in the ns-3 script

with open("trafficFile", "w") as text_file:
    text_file.write("1 %d BE UDP DL 1000 CBR 200" % (nVhtStations))
    
exp.reset()
throughput=0
episode_tput=0
episode_reward=0
step=1
action=Rate
selected_time[action]+=1

rl = Ns3AIRL(memblock_key, Env, Act)
ns3Settings = {'nVhtStations': nVhtStations, 'radius': radius, 'Rate': mcs,'simulationTime':50,'m_nIntervals':m_nIntervals}
pro = exp.run(setting=ns3Settings, show_output=True)
#    print("run wifi-ofmda", ns3Settings)

while not rl.isFinish():
    with rl as data:
        if data == None:
            break
        print("Throughput is ",data.env.throughput)

        f.write(str(float(data.env.throughput)))
        f.write(',')
        # store observation
        throughput=data.env.throughput
            
        # compute reward
        if step<numMCS+1:
            mcs_reward[action][0]=throughput/(bandwidth*np.log2(1+snr))
        else:
            mcs_reward[action].append(throughput/(bandwidth*np.log2(1+snr)))
            
        episode_tput+=throughput
        episode_reward+=mcs_reward[action][-1]
            
        # avg reward
        avg_reward[action]=sum(mcs_reward[action])/selected_time[action]
            
        # next action
        if step<9:
            action=step
        else:
            action=next_action(avg_reward,step,selected_time)
                
        data.act.next_rate = action;
        selected_time[action]+=1
            
        step+=1
pro.wait()
episode_tput/=m_nIntervals;
#    total_tput_vec = np.hstack((total_tput_vec,episode_tput))
#    total_reward_vec = np.hstack((total_reward_vec,episode_reward))
#    print('Total reward after episode {} is finished:{}'.format(i+1,episode_reward))
del exp

f.close

# Save graph
#fig = plt.figure()
#plt.plot(list(range(1,total_episode+1,1)),total_tput_vec,'-')
#figName='result_ucb_radius'+str(radius)+'m_'+str(simTime)+'s_'+str(m_nIntervals)+'step.png'
#fig.savefig(figName)

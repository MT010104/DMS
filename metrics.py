#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd
from keras import Model
from tqdm import tqdm
from keras.models import load_model
from scipy import stats
from functools import reduce


class nbc(object):
    def __init__(self,train,input,layers,std=0):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.lst=[]
        self.upper=[]
        self.lower=[]
        index_lst=[]

        for index,l in layers:
            self.lst.append(Model(inputs=input,outputs=l))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(train).reshape(len(train),l.shape[-1])
            self.upper.append(np.max(temp,axis=0)+std*np.std(temp,axis=0))
            self.lower.append(np.min(temp,axis=0)-std*np.std(temp,axis=0))
        self.upper=np.concatenate(self.upper,axis=0)
        self.lower=np.concatenate(self.lower,axis=0)
        self.neuron_num=self.upper.shape[0]
        self.lst=list(zip(index_lst,self.lst))

    def rank_fast(self,test,use_lower=False):
        self.neuron_activate=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate=np.concatenate(self.neuron_activate,axis=1)

        upper=(self.neuron_activate>self.upper)
        lower=(self.neuron_activate<self.lower)


        subset=[]
        lst=list(range(len(test)))
        initial=np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num=np.sum(upper[initial])
        if use_lower:
            max_cover_num+=np.sum(lower[initial])
        cover_last_1=upper[initial]
        if use_lower:
            cover_last_2=lower[initial]
        while True:
            flag=False
            for index in tqdm(lst):
                temp1=np.bitwise_or(cover_last_1,upper[index])
                cover1=np.sum(temp1)
                if use_lower:
                    temp2=np.bitwise_or(cover_last_2,lower[index])
                    cover1+=np.sum(temp2)
                if cover1>max_cover_num:
                    max_cover_num=cover1
                    max_index=index
                    flag=True
                    max_cover1=temp1
                    if use_lower:
                        max_cover2=temp2
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_last_1=max_cover1
            if use_lower:
                cover_last_2=max_cover2
            # print(max_cover_num)
        return subset

    def rank_2(self,test,use_lower=False):
        self.neuron_activate=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate=np.concatenate(self.neuron_activate,axis=1)
        if use_lower:
            return np.argsort(np.sum(self.neuron_activate>self.upper,axis=1)+np.sum(self.neuron_activate<self.lower,axis=1))[::-1]
        else:
            return np.argsort(np.sum(self.neuron_activate>self.upper,axis=1))[::-1]



class tknc(object):
    def __init__(self,test,input,layers,k=2):
        self.train=test
        self.input=input
        self.layers=layers
        self.k=k
        self.lst=[]
        self.neuron_activate=[]
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(test).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            print(temp.shape)
            self.neuron_activate.append(temp)
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))

    def rank(self, test):

        neuron = []
        layers_num = 0
        for neu in self.neuron_activate:
            neuron.append(np.argsort(neu, axis=1)[:, -self.k:]+layers_num)
            layers_num += neu.shape[-1]
        neuron = np.concatenate(neuron, axis=1)

        subset = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover = len(np.unique(neuron[initial]))
        cover_now = neuron[initial]

        while True:
            flag = False
            for index in tqdm(lst):
                temp = np.union1d(cover_now, neuron[index])
                cover1 = len(temp)
                if cover1 > max_cover:
                    max_cover = cover1
                    max_index = index
                    flag = True
                    max_cover_now = temp
            if not flag or len(lst) == 1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_now = max_cover_now
        if len(subset) < len(lst):
            np.random.shuffle(lst)
            subset.extend(lst)
        return subset

## deepxplore
class nac(object):
    def __init__(self, test, input, layers, t=0):
        self.train = test
        self.input = input
        self.layers = layers
        self.t = t
        self.lst = []
        self.neuron_activate = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(test).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            temp = 1/(1+np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        self.neuron_num = self.neuron_activate.shape[-1]
        self.lst = list(zip(index_lst, self.lst))

    def rank_fast(self, test):
        upper = (self.neuron_activate > self.t)

        subset = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num = np.sum(upper[initial])
        cover_last_1 = upper[initial]

        while True:
            flag = False
            for index in tqdm(lst):
                temp1 = np.bitwise_or(cover_last_1, upper[index])
                cover1 = np.sum(temp1)
                if cover1 >= max_cover_num:
                    max_cover_num = cover1
                    max_index = index
                    flag = True
                    max_cover1 = temp1
            if not flag or len(lst) == 1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_last_1 = max_cover1
            # print(max_cover_num)

        return subset

    def rank_2(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            temp = 1/(1+np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate,axis=1)
        return np.argsort(np.sum(self.neuron_activate > self.t, axis=1))[::-1]

import os
import scipy.io as sio
import random
from random import shuffle
import numpy as np
import h5py
from collections import Counter
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DataLoader
import math
import operator


def data_normalizer(data_30secs):
    data_ch_norm_list=[]
    num_of_channels=data_30secs.shape[1]
    for ch in range(num_of_channels):
        data_ch=np.array(data_30secs[:,ch,:])
        data_ch_mean=(np.mean(data_ch,axis=1)).reshape(-1,1)
        data_ch_std=(np.std(data_ch,axis=1)).reshape(-1,1)
        if math.isnan(data_ch_std) or not data_ch_std:
            data_ch_norm=(data_ch-data_ch_mean)
        else:
            data_ch_norm=(data_ch-data_ch_mean)/data_ch_std
        data_ch_norm_list.append(data_ch_norm[0])
    x=np.array([data_ch_norm_list])
    return x 

class my_generator1(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        f.close()
        print("Total Length instances in generator:",len_instances)
        return len_instances

    def __getitem__(self, idx):
        f = h5py.File(self.hdf5_file, 'r')
        x_30sec_epoch=f['data'][idx]
        y_30sec_epoch=f['label'][idx][0]
        x_30sec_epoch=data_normalizer(x_30sec_epoch)
        f.close()
        return idx, x_30sec_epoch, y_30sec_epoch



def classes_counting(path_to_matfiles_folder):
    mat_files = []
    for i in os.listdir(path_to_matfiles_folder):
        mat_files.append(i)
    mat_files.sort()
    #print(mat_files)
    count = np.zeros(5)
    for i in range(len(mat_files)):
        mat = sio.loadmat(path_to_matfiles_folder+mat_files[i])
        #print(mat_files[i])
        y=np.reshape(mat['signals']['annotations'][0][0][0],((-1,time_period_sample)))
        for label in y:
            for k in range(5):
                label=np.unique(label,axis=0)
                if label== k:
                    count[k] += 1
    return count

def class_distribution(hdf5_file):
    f = h5py.File(hdf5_file, 'r')
    dic_class=dict(Counter(np.reshape(f['label'],(-1))))
    f.close()
    print("dic_class:",dic_class)
    sum_class_dist=dic_class[0]+dic_class[1]+dic_class[2]+dic_class[3]+dic_class[4]
    weight=torch.tensor([1-(dic_class[0]/sum_class_dist),1-(dic_class[1]/sum_class_dist),1-(dic_class[2]/sum_class_dist),1-(dic_class[3]/sum_class_dist),1-(dic_class[4]/sum_class_dist)])
    print("weight:",weight)
    #input("halt")
    return dic_class,weight

class MutableSlice(object):
    def __init__(self, baselist, begin, end=None):
        self._base = baselist
        self._begin = begin
        self._end = len(baselist) if end is None else end

    def __len__(self):
        return self._end - self._begin

    def __getitem__(self, i):
        return self._base[self._begin + i]

    def __setitem__(self, i, val):
        self._base[i + self._begin] = val


class CustomRandomSamplerSlicedShuffled(Sampler):
    def __init__(self, hdf5_file, dic_length):
        self.hdf5_file = hdf5_file
        self.dic_length=dic_length
    
    def __iter__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        OriginalIndices=list(range(0,len_instances))
        #print(self.dic_length.items())
        for item in self.dic_length.items():
            if item[0]==0:
                begin=0
            else:
                begin=self.dic_length[item[0]-1]
            end=item[1]
            slicedIndices=MutableSlice(OriginalIndices,begin,end)
            random.shuffle(slicedIndices)
        iter_shuffledIndex=iter(OriginalIndices)
        f.close()
        return iter_shuffledIndex
    
    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        f.close()
        return len_instances

class CustomRandomBatchSamplerSlicedShuffled(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
    
    def __iter__(self):
        batch = []
        len_tillCurrent=0
        file_current=0
        batch_no=0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent=len_tillCurrent+1
            if len(batch) == self.batch_size or len_tillCurrent==self.file_length_dic[str(file_current)]:
                batch_no=batch_no+1
                print("batch:",batch_no)
                #batch.sort()
                yield batch
                batch = []
            if len_tillCurrent==self.file_length_dic[str(file_current)]:
                len_tillCurrent = 0
                file_current=file_current+1
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.file_length_dic.values()))/self.batch_size),dtype='int32')
        print("length in batch sampler:",length)
        return length

class CustomWeightedRandomSamplerSlicedShuffled(Sampler):
    def __init__(self, hdf5_file, dic_length):
        self.hdf5_file = hdf5_file
        self.dic_length = dic_length
    
    def __iter__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances=f['data'].shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count, weights=class_distribution(self.hdf5_file)
        original_class_count=np.array([class_count[0],class_count[1],class_count[2],class_count[3],class_count[4]])
        max_class=max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1],class_count[2],class_count[3],class_count[4]])
        labels=f['label'][:len_instances].reshape(-1)
        diff_count_class_new = np.zeros(5)
        for i in range(0,5):
            if i!=max_class[0]:
                ratio_diff_of_this_class = np.int(np.floor(diff_count_class[i]/original_class_count[i]))
                diff_count_class_new[i] = diff_count_class[i]-ratio_diff_of_this_class*original_class_count[i]
                for j in range(ratio_diff_of_this_class):
                    OriginalIndices=np.append(OriginalIndices,np.where(labels==i)[0],axis=0)
        for i in range(0,5):
            if i!=max_class[0]:
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],np.int(diff_count_class_new[i])),axis=0)
        OriginalIndices=np.sort(OriginalIndices)
        for item in self.dic_length.items():
            if item[0]==0:
                #begin_value=0
                begin_index=0
            else:
                begin_value=self.dic_length[item[0]-1]
                begin_index=np.where(OriginalIndices==begin_value)[0][0]
            end_index=np.where(OriginalIndices==item[1]-1)[0][-1]
            slicedIndices=MutableSlice(OriginalIndices,begin_index,end_index+1)
            random.shuffle(slicedIndices)
        iter_shuffledIndex=iter(OriginalIndices)
        f.close()
        return iter_shuffledIndex
    
    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        class_count,weights=class_distribution(self.hdf5_file)
        max_class=max(class_count.items(), key=operator.itemgetter(1))
        f.close()
        len_instances_oversample=max_class[1]*5
        print("Sampler __len__:",len_instances_oversample)
        return len_instances_oversample
        
class CustomWeightedRandomBatchSamplerSlicedShuffled(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
    
    def __iter__(self):
        batch = []
        file_current=0
        batch_no=0
        for idx in self.sampler:
            if len(batch)>0 and idx>=self.file_length_dic[file_current]:
                batch_no=batch_no+1
                print("batch:",batch_no)
                yield batch
                batch = []
                file_current=file_current+1
                print("file no __iter__:",file_current)
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch_no=batch_no+1
                print("batch:",batch_no)
                #batch.sort()
                yield batch
                batch = []  
                
        if len(batch)>0:
            batch_no=batch_no+1
            print("batch:",batch_no)
            yield batch
    
    def __len__(self):
        iterator=self.sampler
        sampler_array=np.array(list(iterator))
        totalBatches=0
        for len_item in self.file_length_dic.items():
            if len_item[0]==0:
                previous_file_len=0
            else:
                previous_file_len=self.file_length_dic[len_item[0]-1]
            count_oversampled_oneFile=((previous_file_len<sampler_array) & (sampler_array<len_item[1])).sum()
            totalBatches=totalBatches+np.ceil(count_oversampled_oneFile/self.batch_size)
        print("batch sampler length:",totalBatches)
        return totalBatches
        
class CustomSequentialLSTMBatchSampler_ReturnAllChunks(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic,seq_length):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic
    
    def __iter__(self):
        batch = []
        len_tillCurrent=0
        file_current=0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent=len_tillCurrent+1
            if len(batch) == self.batch_size or len_tillCurrent==self.file_length_dic[str(file_current)]:
                if len_tillCurrent==self.file_length_dic[str(file_current)]:
                    if len(batch)%self.seq_length!=0:
                        quoteint=int(len(batch)/self.seq_length)
                        remainder=self.seq_length*(quoteint+1)-len(batch)
                        batch.extend(range(idx-len_tillCurrent+1,idx-len_tillCurrent+1+remainder))
                    len_tillCurrent = 0
                    file_current=file_current+1
                yield batch
                batch = []
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.file_length_dic.values()))/self.batch_size),dtype='int32')
        print(length)
        return length

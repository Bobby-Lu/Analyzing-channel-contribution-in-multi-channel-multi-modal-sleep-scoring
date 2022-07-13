import scipy.io as sio
import numpy as np
import torch
from collections import OrderedDict
import os
import math
import cnn_can_models

def data_normalizer(allData_30secs):
    data_shape=allData_30secs.shape
    x=np.empty(data_shape)
    m=0
    for data in allData_30secs:
        data_ch_norm_list=[]
        for ch in range(data.shape[1]):
            data_ch=np.array(data[:,ch])
            data_ch_mean=(np.mean(data_ch,axis=1)).reshape(-1,1)
            data_ch_std=(np.std(data_ch,axis=1)).reshape(-1,1)
            if math.isnan(data_ch_std) or not data_ch_std:
                data_ch_norm=(data_ch-data_ch_mean)
            else:
                data_ch_norm=(data_ch-data_ch_mean)/data_ch_std
            data_ch_norm_list.append(data_ch_norm[0])
        x[m]=np.array([data_ch_norm_list])
        m=m+1
    x=torch.from_numpy(x).float()
    return x

def load_model(path):
    model=cnn_can_models.SleepMultiChannelNet(lstm_option=False)
    state_dict_new_model=model.state_dict()
    checkpoint = torch.load(path,map_location=torch.device('cpu') )
    state_dict_pretrained=checkpoint['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in state_dict_pretrained.items():
        state_dict_remove_module[k] = v
    state_dict_new_model.update(state_dict_remove_module)
    model.load_state_dict(state_dict_new_model)
    return model


def predict(path,data):
    model=load_model(path)
    model.eval()
    data=data_normalizer(data)
    output, attn_weights=model(data.to(device))
    predict_class=output.argmax(dim=1, keepdim=True)
    return predict_class, attn_weights

device='cpu' 

mat_files = []

path_to_matfiles_folder='RQ2/SampleData_example/'

for i in os.listdir(path_to_matfiles_folder):
    mat_files.append(i)

mat_files.sort()

channel_attention_concrete = np.zeros([5,5])
count = np.zeros(5)

for k in range(0,60):
    datas = sio.loadmat(path_to_matfiles_folder+mat_files[k])
    print(k)
    annotation = datas['signals'][0][0]['annotation'][0]
    num = annotation.size//3750
    annotations = []
    for i in range(0,num):
        signal1 = datas['signals'][0][0]['eeg1'][0][i*3750:(i+1)*3750]
        signal2 = datas['signals'][0][0]['eeg2'][0][i*3750:(i+1)*3750]
        signal3 = datas['signals'][0][0]['eogL'][0][i*3750:(i+1)*3750]
        signal4 = datas['signals'][0][0]['eogR'][0][i*3750:(i+1)*3750]
        signal5 = datas['signals'][0][0]['emg'][0][i*3750:(i+1)*3750]
        signals = np.array([[signal1,signal2,signal3,signal4,signal5]])
        data = []
        data.append(signals)
        annotations.append(datas['signals'][0][0]['annotation'][0][i*3750])
        #print(signal1.shape)
        data = np.array(data)
        #data.shape
        data = torch.from_numpy(data)
        data=data.float()

        output_class, attention_weight=predict('RQ2/eCAN/cnn_can_weightedloss.tar',data)
        #print(output_class)
        #print(attention_weight)
        for j in range(0,5):
            if output_class==j:
                #print(output_class.shape[0])
                count[j] += 1
                channel_attention_concrete[j] += attention_weight.detach().numpy()[0]

for j in range(0,5):
    channel_attention_concrete[j] = channel_attention_concrete[j]/count[j]
print(channel_attention_concrete) 
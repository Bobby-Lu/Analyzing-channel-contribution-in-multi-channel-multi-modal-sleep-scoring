import scipy.io as sio
import numpy as np
import torch
from collections import OrderedDict
import os
import math
import torch.nn as nn

Fs = 125
time_period_sample = 3750

class CNNBlock(nn.Module):
    def __init__(self, sub_channel):
        super(CNNBlock, self).__init__()
        self.dir1 = nn.Sequential(
            nn.Conv1d(sub_channel, 64, round(Fs/2), groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(32),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 8, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8)
        )
        self.dir2 = nn.Sequential(
            nn.Conv1d(sub_channel, 64, 4*Fs, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(32),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 8, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, groups=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8)
        )
        self.dropout = nn.Dropout(p=0.5)
        self.apply(self.init_weights)
    def init_weights(self,x):
        if type(x) == nn.Conv1d or type(x) == nn.BatchNorm1d:
            x.weight.data.normal_(0, 0.05) 
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    
    def forward(self, inputs):
        x1 = self.dir1(inputs)
        #print(x1.shape)
        x2 = self.dir2(inputs)
        #print(x2.shape)
        x1 = x1.view(-1,self.num_flat_features(x1))
        x2 = x2.view(-1,self.num_flat_features(x2))
        x = torch.cat((x1,x2),1)
        x = self.dropout(x)
        return x
    
class SleepMultiChannelNet(nn.Module):
    def __init__(self, lstm_option):
        super(SleepMultiChannelNet, self).__init__()
        self.CNNBlocks = nn.ModuleDict()
        self.lstm_option = lstm_option
        self.CNNBlocks['eeg1'] = CNNBlock(1)
        self.CNNBlocks['eeg2'] = CNNBlock(1)
        self.CNNBlocks['eog1'] = CNNBlock(1)
        self.CNNBlocks['eog2'] = CNNBlock(1)
        self.CNNBlocks['emg'] = CNNBlock(1)
        self.linear = nn.Linear(6720,5)
        self.init_weights()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.05)
    def forward(self, x):
        activations = []
        cnn_eeg1 = self.CNNBlocks['eeg1'](x[:,:,0:1,:].view(-1,1,time_period_sample))
        cnn_eeg2 = self.CNNBlocks['eeg2'](x[:,:,1:2,:].view(-1,1,time_period_sample))
        cnn_eog1 = self.CNNBlocks['eog1'](x[:,:,2:3,:].view(-1,1,time_period_sample))
        cnn_eog2 = self.CNNBlocks['eog2'](x[:,:,3:4,:].view(-1,1,time_period_sample))
        cnn_emg = self.CNNBlocks['emg'](x[:,:,4:5,:].view(-1,1,time_period_sample))
        out_features = torch.cat((cnn_eeg1,cnn_eeg2,cnn_eog1,cnn_eog2,cnn_emg),1)
        activations.append(out_features.detach().numpy())
        if self.lstm_option == True:
            outputs = out_features
        else:
            outputs = self.linear(out_features)
        activations.append(outputs.detach().numpy())
        return outputs,activations






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
    model=SleepMultiChannelNet(lstm_option=False)
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
    output, activations=model(data.to(device))
    predict_class=output.argmax(dim=1, keepdim=True)
    return predict_class, activations






mat_files = []
device="cpu"

path_to_matfiles_folder='/RQ2/SampleData_example/'

for i in os.listdir(path_to_matfiles_folder):
    mat_files.append(i)

mat_files.sort()

for j in range(len(mat_files)):
    print(j)
    datas = sio.loadmat(path_to_matfiles_folder+mat_files[j].strip('._'))

    annotation = datas['signals'][0][0]['annotation'][0]
    num = annotation.size//3750
    annotations = []
    output_classes = []
    activations1 = []
    activations2 = []
    
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
        data = torch.tensor(data)
        data=data.float()

        output_class, activation=predict('/RQ2/LRP/withoutSL/cnn_weightedloss.tar',data)
        output_classes.append(output_class[0])
        activations1.append(activation[0])
        activations2.append(activation[1])
        
    output_classes = np.asarray(output_classes)
    activations1 = np.asarray(activations1)
    activations2 = np.asarray(activations2)
    np.save('RQ2/LRP/Predictions/prediction'+str(j)+'.npy',output_classes)
    np.save('RQ2/LRP/Activations/activation'+str(j)+'_1.npy',activations1)
    np.save('RQ2/LRP/Activations/activation'+str(j)+'_2.npy',activations2)
#activations
#[EEG:[input,l2conv,l2max,l3conv,l3max,flatten],EOG,EMG,linear,output]
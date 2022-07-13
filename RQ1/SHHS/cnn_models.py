import torch
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
        self.CNNBlocks['eeg'] = CNNBlock(2)
        self.CNNBlocks['eog'] = CNNBlock(2)
        self.CNNBlocks['emg'] = CNNBlock(1)
        self.linear = nn.Linear(4032,5)
        self.init_weights()
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.05)
    def forward(self, x):
        cnn_eeg = self.CNNBlocks['eeg'](x[:,:,0:2,:].view(-1,2,time_period_sample))
        cnn_eog = self.CNNBlocks['eog'](x[:,:,2:4,:].view(-1,2,time_period_sample))
        cnn_emg = self.CNNBlocks['emg'](x[:,:,4:5,:].view(-1,1,time_period_sample))
        out_features = torch.cat((cnn_eeg,cnn_eog,cnn_emg),1)
        if self.lstm_option == True:
            outputs = out_features
        else:
            outputs = self.linear(out_features)
        return outputs



import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, ):
        super(CNNBlock, self).__init__()
        self.dir1 = nn.Sequential(
            nn.Conv1d(1, 64, round(Fs/2), groups=1),
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
            nn.Conv1d(1, 64, 4*Fs, groups=1),
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
        if type(x) == nn.Conv1d or type(x) == nn.BatchNorm1d or type(x) == nn.Linear:
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
    
class ChannelAttention(nn.Module):
    def __init__(self,):
        super(ChannelAttention, self).__init__()
        self.ca_layer_encode = nn.Linear(5,16)
        self.ca_layer_decode = nn.Linear(16,5)
        self.ReLU = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.ca_layer_encode.weight.data.normal_(0, 0.05)
        self.ca_layer_decode.weight.data.normal_(0, 0.05)
    def forward(self, cnn_features):
        x = F.avg_pool1d(cnn_features,1344).view(-1,5)
        x = self.ca_layer_encode(x)
        x = self.ReLU(x)
        x = self.ca_layer_decode(x)
        attn_weights = F.softmax(x,dim=1)
        return attn_weights
        
class SleepMultiChannelNet(nn.Module):
    def __init__(self, lstm_option):
        super(SleepMultiChannelNet, self).__init__()
        self.CNNBlocks = nn.ModuleDict()
        self.lstm_option = lstm_option
        self.CNNBlocks['eeg1'] = CNNBlock()
        self.CNNBlocks['eeg2'] = CNNBlock()
        self.CNNBlocks['eog1'] = CNNBlock()
        self.CNNBlocks['eog2'] = CNNBlock()
        self.CNNBlocks['emg'] = CNNBlock()
        self.CAModule = ChannelAttention()
        self.linear = nn.Linear(6720,5)
        self.init_weights()
        
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.05)
        
    def forward(self, x):
        cnn_eeg1 = self.CNNBlocks['eeg1'](x[:,:,0:1,:].view(-1,1,time_period_sample))
        cnn_eeg2 = self.CNNBlocks['eeg2'](x[:,:,1:2,:].view(-1,1,time_period_sample))
        cnn_eog1 = self.CNNBlocks['eog1'](x[:,:,2:3,:].view(-1,1,time_period_sample))
        cnn_eog2 = self.CNNBlocks['eog2'](x[:,:,3:4,:].view(-1,1,time_period_sample))
        cnn_emg = self.CNNBlocks['emg'](x[:,:,4:5,:].view(-1,1,time_period_sample))
        attn_weights = self.CAModule(torch.cat((cnn_eeg1.view(-1,1,1344),cnn_eeg2.view(-1,1,1344),cnn_eog1.view(-1,1,1344),cnn_eog2.view(-1,1,1344),cnn_emg.view(-1,1,1344)),1))
        #print(attn_weights.shape)
        x = torch.cat((attn_weights[:,0].view(-1,1)*F.softmax(cnn_eeg1,dim=1),attn_weights[:,1].view(-1,1)*F.softmax(cnn_eeg2,dim=1),attn_weights[:,2].view(-1,1)*F.softmax(cnn_eog1,dim=1),attn_weights[:,3].view(-1,1)*F.softmax(cnn_eog2,dim=1),attn_weights[:,4].view(-1,1)*F.softmax(cnn_emg,dim=1)),1)
        x = self.linear(x)
        return x,attn_weights



import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import utils
import cnn_models
import lstm_models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,SequentialSampler
import os
import torch.nn as nn
import torch
import pickle
import math
from collections import OrderedDict
from openpyxl import Workbook
import openpyxl as op
import torch.nn.functional as F

def load_pretrained_model_for_LSTM(model,path):
    state_dict_new_model=model.state_dict()
    checkpoint = torch.load(path,map_location=torch.device('cpu'))
    state_dict_pretrained=checkpoint['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in state_dict_pretrained.items():
        if k!='linear.weight' and k!='linear.bias':
            #k = k[7:] # remove `module.`
            state_dict_remove_module[k] = v
    state_dict_new_model.update(state_dict_remove_module)
    model.load_state_dict(state_dict_new_model)
    return model

def loss_fn():
    criterion = nn.CrossEntropyLoss()
    return criterion

def conf_mat_create(predicted,true,correct,total_30sec_epochs,conf_mat):
    total_30sec_epochs+=true.size()[0]
    correct += predicted.eq(true.view_as(predicted)).sum().item()
    conf_mat=conf_mat+confusion_matrix(true.cpu().numpy(),predicted.cpu().numpy(),labels=classes)
    return correct, total_30sec_epochs,conf_mat


def test(cnn_model, lstm_model, data_test, epoch):
    cnn_model.eval()
    lstm_model.eval()
    total_30sec_epochs_test = 0
    test_loss = 0.0
    correct_test = 0
    con_mat_test=np.zeros((5,5))
    lossfn1=loss_fn()
    file_no=0
    count_file_len=0
    for test_idx, test_batch, test_labels in data_test:
        test_batch, test_labels=test_batch.to(device), test_labels.to(device)
        cnn_features = cnn_model(test_batch)
        decoder_input = F.one_hot(test_labels % 5, num_classes=5).type(torch.FloatTensor).to(device)
        if test_idx[0]==count_file_len:
            count_file_len+=file_length_dic_test[str(file_no)]
            file_no+=1
            print("New Patient")
            decoder_input_first = F.one_hot(torch.tensor(0) % 5, num_classes=5).type(torch.FloatTensor).to(device)
        else:
            decoder_input_first = decoder_input_last
        output, decoder_input_last = lstm_model(decoder_input_first, decoder_input, cnn_features)
        test_labels_crop = test_labels.view(-1)
        test_pred = output.argmax(dim=1,keepdim=True)
        if max(test_idx)!=test_idx[-1]:
            idx_nump=test_idx.numpy()
            max_idx=np.where(idx_nump==max(idx_nump))[0][0]
            print(max_idx)
            test_pred=test_pred[:max_idx+1]
            test_labels_crop=test_labels_crop[:max_idx+1]
            output=output[:max_idx+1]
        loss1 = lossfn1(output,test_labels_crop).item()
        test_loss += test_labels_crop.size()[0]*loss1
        correct_test,total_30sec_epochs_test,con_mat_test=conf_mat_create(test_pred,test_labels_crop,correct_test,total_30sec_epochs_test,con_mat_test)
    print("conf_mat_test:",con_mat_test)
    print("total_30sec_epochs_test:",total_30sec_epochs_test)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch:{}\n'.format(test_loss/total_30sec_epochs_test, correct_test, total_30sec_epochs_test,100. * correct_test / total_30sec_epochs_test,epoch+1))
    sheet1.append([0,1,2,3,4])
    for row in con_mat_test.tolist():
        sheet1.append(row)


if __name__ == '__main__': 
    batch_size = 24
    classes=[0,1,2,3,4]
    epochs = 200
    seq_length = 8
    path_to_cnn_model = '/RQ1/SHHS/models/cnn_weightedloss.tar'
    path_to_lstm_model = '/RQ1/SHHS/models/lstm_weightedloss.tar'

    path_to_file_length_test='/SHHS/testFilesNum30secEpochs_all_SHHS.pkl'
    f_file_length_test=open(path_to_file_length_test,'rb')
    file_length_dic_test=pickle.load(f_file_length_test)
    f_file_length_test.close()
    
    path_to_hdf5_file_test = '/SHHS/hdf5_file_test_all_chunking_SHHS.hdf5'
    path_to_results = '/RQ1/SHHS/test_lstm.xlsx'
    
    if os.path.isfile(path_to_results):
        wb = op.load_workbook(path_to_results)
        sheet1 = wb.get_sheet_by_name('Sheet 1')
    else:
        wb=Workbook()
        sheet1=wb.active
        sheet1.title = "Sheet 1"
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    cnn_model = cnn_models.SleepMultiChannelNet(lstm_option=True) 
    lstm_model = lstm_models.SleepMultiChannelNet(input_size=4032,hidden_size=256,output_size=5,seq_length=seq_length,device=device)

    cnn_model = load_pretrained_model_for_LSTM(cnn_model,path_to_cnn_model)
    lstm_model.load_state_dict(torch.load(path_to_lstm_model,map_location=torch.device('cpu'))['state_dict'])
    cnn_model.to(device)
    lstm_model.to(device)

    data_gen_test=utils.my_generator1(path_to_hdf5_file_test)
    sampler_test=SequentialSampler(data_gen_test)
    batch_sampler_test=utils.CustomSequentialLSTMBatchSampler_ReturnAllChunks(sampler_test,batch_size*seq_length,file_length_dic_test,seq_length)
    data_test=DataLoader(data_gen_test,batch_size=1,batch_sampler=batch_sampler_test)
    print("test dataset loaded") 
     
    test(cnn_model, lstm_model,data_test,epochs)
    wb.save(path_to_results)
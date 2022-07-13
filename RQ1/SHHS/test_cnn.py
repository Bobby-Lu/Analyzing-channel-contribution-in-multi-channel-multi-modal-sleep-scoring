import h5py    
import numpy as np  
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from sklearn.metrics import accuracy_score
from cnn_models import SleepMultiChannelNet
import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch
from collections import OrderedDict
from openpyxl import Workbook
import openpyxl as op

def loss_fn():
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    return criterion

def conf_mat_create(predicted,true,correct,total_30sec_epochs,conf_mat):
    total_30sec_epochs+=true.size()[0]
    correct += predicted.eq(true.view_as(predicted)).sum().item()
    conf_mat=conf_mat+confusion_matrix(true.cpu().numpy(),predicted.cpu().numpy(),labels=classes)
    return correct, total_30sec_epochs,conf_mat

def load_model(model,path):
    checkpoint = torch.load(path)
    state_dict=checkpoint['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in state_dict.items():
        #k = k[7:] # remove `module.`
        state_dict_remove_module[k] = v
    model.load_state_dict(state_dict_remove_module)
    return model

def test(model, data_test, epoch):
    model.eval()
    total_30sec_epochs_test = 0
    test_loss = 0.0
    correct_test = 0
    con_mat_test=np.zeros((5,5))
    count = 0
    for test_idx, test_data, test_labels in data_test:
        test_data, test_labels=test_data.to(device), test_labels.to(device)
        output = model(test_data)
        test_labels_crop = test_labels.view(-1)
        test_pred = output.argmax(dim=1,keepdim=True)
        correct_test,total_30sec_epochs_test,con_mat_test=conf_mat_create(test_pred,test_labels_crop,correct_test,total_30sec_epochs_test,con_mat_test)

    print("conf_mat_test:",con_mat_test)
    print("total_30sec_epochs_test:",total_30sec_epochs_test)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch:{}\n'.format(total_30sec_epochs_test, correct_test, total_30sec_epochs_test,100. * correct_test / total_30sec_epochs_test,epoch+1))
    sheet1.append([0,1,2,3,4])
    for row in con_mat_test.tolist():
        sheet1.append(row)
        
if __name__ == '__main__': 
    batch_size = 192
    classes=[0,1,2,3,4]
    epochs = 200
    if torch.cuda.device_count() > 1:
        multiple_gpu=True
    else:
        multiple_gpu=False
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda)
    path_to_model = '/RQ1/SHHS/models/cnn_weightedloss.tar'
    path_to_hdf5_file_test = '/SHHS/hdf5_file_test_all_chunking_SHHS.hdf5'
    path_to_results = '/RQ1/SHHS/test_cnn.xlsx'
    
    if os.path.isfile(path_to_results):
        wb = op.load_workbook(path_to_results)
        sheet1 = wb['Sheet 1']
    else:
        wb=Workbook()
        sheet1=wb.active
        sheet1.title = "Sheet 1"

    dataset_test = utils.my_generator1(path_to_hdf5_file_test)
    data_test = DataLoader(dataset_test, num_workers=8, batch_size=192)

    model = SleepMultiChannelNet(lstm_option=False) 
    model = load_model(model, path_to_model)
    model.to(device)

    test(model,data_test,epoch=200)
    wb.save(path_to_results)
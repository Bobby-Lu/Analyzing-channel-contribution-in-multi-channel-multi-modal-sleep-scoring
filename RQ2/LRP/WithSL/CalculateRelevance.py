import torch
import numpy as np
from LRP import Convolution
from LRP import Convolution_first_layer
from LRP import MaxPool
from LRP import Flatten
from LRP import Linear
import cnn_models
from collections import OrderedDict
from scipy.special import softmax
import gc 


def load_model(path):
    model=cnn_models.SleepMultiChannelNet(False)
    state_dict_new_model=model.state_dict()
    checkpoint = torch.load(path,map_location=torch.device('cpu') )
    state_dict_pretrained=checkpoint['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in state_dict_pretrained.items():
        state_dict_remove_module[k] = v
    state_dict_new_model.update(state_dict_remove_module)
    model.load_state_dict(state_dict_new_model)
    return model

model = load_model('/RQ2/LRP/WithSL/cnn_weightedloss.tar')

na = np.newaxis
R_wake = np.zeros(5)
count1 = 0
R_N1 = np.zeros(5)
count2 = 0
R_N2 = np.zeros(5)
count3 = 0
R_N3 = np.zeros(5)
count4 = 0
R_REM = np.zeros(5)
count5 = 0
for i in range(0,60):
    
    predictions = np.load('/RQ2/LRP/WithSL/Predictions/prediction'+str(i)+'.npy',allow_pickle=True)
    activations = np.load('/RQ2/LRP/WithSL/Activations/activation'+str(i)+'.npy',allow_pickle=True)

    activations_output= softmax(activations[4],axis=1)

    #Linear Layer
    activations_linear = activations[3]
    weights_linear = model.linear.weight.data.numpy().T
    bias_linear = model.linear.bias.data.numpy()
    linear = Linear(4032,5,activations_linear,weights_linear,bias_linear)
    R = linear.lrp(activations_output)

    #concatenation
    R_EEG_dir1 = R[:,:704]
    R_EEG_dir2 = R[:,704:1344]
    R_EOG_dir1 = R[:,1344:2048]
    R_EOG_dir2 = R[:,2048:2688]
    R_EMG = R[:,2688:4032]
    
    #EEG_dir1
    #Flatten Layer
    activations_EEG_dir1_flatten = activations[0][1][5][:,:,na,:]
    EEG_dir1_flatten = Flatten(activations_EEG_dir1_flatten)
    R_EEG_dir1 = EEG_dir1_flatten.lrp(R_EEG_dir1)
    #Maxpool2
    activations_EEG_dir1_maxpool2 = activations[0][1][4][:,:,na,:]
    EEG_dir1_maxpool2 = MaxPool(activations_EEG_dir1_maxpool2,pool=(1,8),stride=(1,8))
    R_EEG_dir1 = EEG_dir1_maxpool2.lrp(R_EEG_dir1)
    #Conv3
    activations_EEG_dir1_conv3 = activations[0][1][3][:,:,na,:]
    weights_EEG_dir1_conv3 = model.CNNBlocks['eeg'].dir1[11].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir1_conv3 = model.CNNBlocks['eeg'].dir1[11].bias.data.numpy()
    EEG_dir1_conv3 = Convolution(activations_EEG_dir1_conv3,weights_EEG_dir1_conv3,bias_EEG_dir1_conv3,filtersize=(1,8,64,64),stride=(1,1))
    R_EEG_dir1 = EEG_dir1_conv3.lrp(R_EEG_dir1)
    #Conv2
    activations_EEG_dir1_conv2 = activations[0][1][2][:,:,na,:]
    weights_EEG_dir1_conv2 = model.CNNBlocks['eeg'].dir1[8].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir1_conv2 = model.CNNBlocks['eeg'].dir1[8].bias.data.numpy()
    EEG_dir1_conv2 = Convolution(activations_EEG_dir1_conv2,weights_EEG_dir1_conv2,bias_EEG_dir1_conv2,filtersize=(1,8,64,64),stride=(1,1))
    R_EEG_dir1 = EEG_dir1_conv2.lrp(R_EEG_dir1)
    #Conv1
    activations_EEG_dir1_conv1 = activations[0][1][1][:,:,na,:]
    weights_EEG_dir1_conv1 = model.CNNBlocks['eeg'].dir1[5].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir1_conv1 = model.CNNBlocks['eeg'].dir1[5].bias.data.numpy()
    EEG_dir1_conv1 = Convolution(activations_EEG_dir1_conv1,weights_EEG_dir1_conv1,bias_EEG_dir1_conv1,filtersize=(1,8,64,64),stride=(1,1))
    R_EEG_dir1 = EEG_dir1_conv1.lrp(R_EEG_dir1)
    #Maxpool1
    activations_EEG_dir1_maxpool1 = activations[0][1][0][:,:,na,:]
    EEG_dir1_maxpool1 = MaxPool(activations_EEG_dir1_maxpool1,pool=(1,32),stride=(1,32))
    R_EEG_dir1 = EEG_dir1_maxpool1.lrp(R_EEG_dir1)
    #Conv
    activations_EEG_dir1_conv = activations[0][0][:,:,na,:]
    weights_EEG_dir1_conv = model.CNNBlocks['eeg'].dir1[0].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir1_conv = model.CNNBlocks['eeg'].dir1[0].bias.data.numpy()
    EEG_dir1_conv = Convolution_first_layer(activations_EEG_dir1_conv,weights_EEG_dir1_conv,bias_EEG_dir1_conv,filtersize=(1,62,2,64),stride=(1,1))
    R_EEG_dir1 = EEG_dir1_conv.lrp(R_EEG_dir1)

    #EEG_dir2
    #Flatten Layer
    activations_EEG_dir2_flatten = activations[0][2][5][:,:,na,:]
    EEG_dir2_flatten = Flatten(activations_EEG_dir2_flatten)
    R_EEG_dir2 = EEG_dir2_flatten.lrp(R_EEG_dir2)
    #Maxpool2
    activations_EEG_dir2_maxpool2 = activations[0][2][4][:,:,na,:]
    EEG_dir2_maxpool2 = MaxPool(activations_EEG_dir2_maxpool2,pool=(1,8),stride=(1,8))
    R_EEG_dir2 = EEG_dir2_maxpool2.lrp(R_EEG_dir2)
    #Conv3
    activations_EEG_dir2_conv3 = activations[0][2][3][:,:,na,:]
    weights_EEG_dir2_conv3 = model.CNNBlocks['eeg'].dir2[11].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir2_conv3 = model.CNNBlocks['eeg'].dir2[11].bias.data.numpy()
    EEG_dir2_conv3 = Convolution(activations_EEG_dir2_conv3,weights_EEG_dir2_conv3,bias_EEG_dir2_conv3,filtersize=(1,8,64,64),stride=(1,1))
    R_EEG_dir2 = EEG_dir2_conv3.lrp(R_EEG_dir2)
    #Conv2
    activations_EEG_dir2_conv2 = activations[0][2][2][:,:,na,:]
    weights_EEG_dir2_conv2 = model.CNNBlocks['eeg'].dir2[8].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir2_conv2 = model.CNNBlocks['eeg'].dir2[8].bias.data.numpy()
    EEG_dir2_conv2 = Convolution(activations_EEG_dir2_conv2,weights_EEG_dir2_conv2,bias_EEG_dir2_conv2,filtersize=(1,8,64,64),stride=(1,1))
    R_EEG_dir2 = EEG_dir2_conv2.lrp(R_EEG_dir2)
    #Conv1
    activations_EEG_dir2_conv1 = activations[0][2][1][:,:,na,:]
    weights_EEG_dir2_conv1 = model.CNNBlocks['eeg'].dir2[5].weight.data.numpy()[:,:,na,:]
    bias_EEG_dir2_conv1 = model.CNNBlocks['eeg'].dir2[5].bias.data.numpy()
    EEG_dir2_conv1 = Convolution(activations_EEG_dir2_conv1,weights_EEG_dir2_conv1,bias_EEG_dir2_conv1,filtersize=(1,8,64,64),stride=(1,1))
    R_EEG_dir2 = EEG_dir2_conv1.lrp(R_EEG_dir2)
    #Maxpool1
    activations_EEG_dir2_maxpool1 = activations[0][2][0][:,:,na,:]
    EEG_dir2_maxpool1 = MaxPool(activations_EEG_dir2_maxpool1,pool=(1,32),stride=(1,32))
    R_EEG_dir2 = EEG_dir2_maxpool1.lrp(R_EEG_dir2)
    np.save('/LRP/R_EEG_dir2/R_EEG_dir2_'+str(i)+'.npy',R_EEG_dir2)
    R_EEG_dir2_save = np.load('/LRP/R_EEG_dir2/R_EEG_dir2_'+str(i)+'.npy')
    for j in range(predictions.shape[0]):
        print(str(i)+'_'+str(j))
        #Conv
        activations_EEG_dir2_conv = activations[0][0][j][na,:,na,:]
        weights_EEG_dir2_conv = model.CNNBlocks['eeg'].dir2[0].weight.data.numpy()[:,:,na,:]
        bias_EEG_dir2_conv = model.CNNBlocks['eeg'].dir2[0].bias.data.numpy()
        EEG_dir2_conv = Convolution_first_layer(activations_EEG_dir2_conv,weights_EEG_dir2_conv,bias_EEG_dir2_conv,filtersize=(1,500,2,64),stride=(1,1))
        R_EEG_dir2 = EEG_dir2_conv.lrp(R_EEG_dir2_save[j][na,:,:,:])

    #EOG_dir1
    #Flatten Layer
    activations_EOG_dir1_flatten = activations[1][1][5][:,:,na,:]
    EOG_dir1_flatten = Flatten(activations_EOG_dir1_flatten)
    R_EOG_dir1 = EOG_dir1_flatten.lrp(R_EOG_dir1)
    #Maxpool2
    activations_EOG_dir1_maxpool2 = activations[1][1][4][:,:,na,:]
    EOG_dir1_maxpool2 = MaxPool(activations_EOG_dir1_maxpool2,pool=(1,8),stride=(1,8))
    R_EOG_dir1 = EOG_dir1_maxpool2.lrp(R_EOG_dir1)
    #Conv3
    activations_EOG_dir1_conv3 = activations[1][1][3][:,:,na,:]
    weights_EOG_dir1_conv3 = model.CNNBlocks['eog'].dir1[11].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir1_conv3 = model.CNNBlocks['eog'].dir1[11].bias.data.numpy()
    EOG_dir1_conv3 = Convolution(activations_EOG_dir1_conv3,weights_EOG_dir1_conv3,bias_EOG_dir1_conv3,filtersize=(1,8,64,64),stride=(1,1))
    R_EOG_dir1 = EOG_dir1_conv3.lrp(R_EOG_dir1)
    #Conv2
    activations_EOG_dir1_conv2 = activations[1][1][2][:,:,na,:]
    weights_EOG_dir1_conv2 = model.CNNBlocks['eog'].dir1[8].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir1_conv2 = model.CNNBlocks['eog'].dir1[8].bias.data.numpy()
    EOG_dir1_conv2 = Convolution(activations_EOG_dir1_conv2,weights_EOG_dir1_conv2,bias_EOG_dir1_conv2,filtersize=(1,8,64,64),stride=(1,1))
    R_EOG_dir1 = EOG_dir1_conv2.lrp(R_EOG_dir1)
    #Conv1
    activations_EOG_dir1_conv1 = activations[1][1][1][:,:,na,:]
    weights_EOG_dir1_conv1 = model.CNNBlocks['eog'].dir1[5].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir1_conv1 = model.CNNBlocks['eog'].dir1[5].bias.data.numpy()
    EOG_dir1_conv1 = Convolution(activations_EOG_dir1_conv1,weights_EOG_dir1_conv1,bias_EOG_dir1_conv1,filtersize=(1,8,64,64),stride=(1,1))
    R_EOG_dir1 = EOG_dir1_conv1.lrp(R_EOG_dir1)
    #Maxpool1
    activations_EOG_dir1_maxpool1 = activations[1][1][0][:,:,na,:]
    EOG_dir1_maxpool1 = MaxPool(activations_EOG_dir1_maxpool1,pool=(1,32),stride=(1,32))
    R_EOG_dir1 = EOG_dir1_maxpool1.lrp(R_EOG_dir1)
    #Conv
    activations_EOG_dir1_conv = activations[1][0][:,:,na,:]
    weights_EOG_dir1_conv = model.CNNBlocks['eog'].dir1[0].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir1_conv = model.CNNBlocks['eog'].dir1[0].bias.data.numpy()
    EOG_dir1_conv = Convolution_first_layer(activations_EOG_dir1_conv,weights_EOG_dir1_conv,bias_EOG_dir1_conv,filtersize=(1,62,2,64),stride=(1,1))
    R_EOG_dir1 = EOG_dir1_conv.lrp(R_EOG_dir1)
  
    #EOG_dir2
    #Flatten Layer
    activations_EOG_dir2_flatten = activations[1][2][5][:,:,na,:]
    EOG_dir2_flatten = Flatten(activations_EOG_dir2_flatten)
    R_EOG_dir2 = EOG_dir2_flatten.lrp(R_EOG_dir2)
    #Maxpool2
    activations_EOG_dir2_maxpool2 = activations[1][2][4][:,:,na,:]
    EOG_dir2_maxpool2 = MaxPool(activations_EOG_dir2_maxpool2,pool=(1,8),stride=(1,8))
    R_EOG_dir2 = EOG_dir2_maxpool2.lrp(R_EOG_dir2)
    #Conv3
    activations_EOG_dir2_conv3 = activations[1][2][3][:,:,na,:]
    weights_EOG_dir2_conv3 = model.CNNBlocks['eog'].dir2[11].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir2_conv3 = model.CNNBlocks['eog'].dir2[11].bias.data.numpy()
    EOG_dir2_conv3 = Convolution(activations_EOG_dir2_conv3,weights_EOG_dir2_conv3,bias_EOG_dir2_conv3,filtersize=(1,8,64,64),stride=(1,1))
    R_EOG_dir2 = EOG_dir2_conv3.lrp(R_EOG_dir2)
    #Conv2
    activations_EOG_dir2_conv2 = activations[1][2][2][:,:,na,:]
    weights_EOG_dir2_conv2 = model.CNNBlocks['eog'].dir2[8].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir2_conv2 = model.CNNBlocks['eog'].dir2[8].bias.data.numpy()
    EOG_dir2_conv2 = Convolution(activations_EOG_dir2_conv2,weights_EOG_dir2_conv2,bias_EOG_dir2_conv2,filtersize=(1,8,64,64),stride=(1,1))
    R_EOG_dir2 = EOG_dir2_conv2.lrp(R_EOG_dir2)
    #Conv1
    activations_EOG_dir2_conv1 = activations[1][2][1][:,:,na,:]
    weights_EOG_dir2_conv1 = model.CNNBlocks['eog'].dir2[5].weight.data.numpy()[:,:,na,:]
    bias_EOG_dir2_conv1 = model.CNNBlocks['eog'].dir2[5].bias.data.numpy()
    EOG_dir2_conv1 = Convolution(activations_EOG_dir2_conv1,weights_EOG_dir2_conv1,bias_EOG_dir2_conv1,filtersize=(1,8,64,64),stride=(1,1))
    R_EOG_dir2 = EOG_dir2_conv1.lrp(R_EOG_dir2)
    #Maxpool1
    activations_EOG_dir2_maxpool1 = activations[1][2][0][:,:,na,:]
    EOG_dir2_maxpool1 = MaxPool(activations_EOG_dir2_maxpool1,pool=(1,32),stride=(1,32))
    R_EOG_dir2 = EOG_dir2_maxpool1.lrp(R_EOG_dir2)
    np.save('/LRP/R_EOG_dir2/R_EOG_dir2_'+str(i)+'.npy',R_EOG_dir2)
    R_EOG_dir2_save = np.load('/LRP/R_EOG_dir2/R_EOG_dir2_'+str(i)+'.npy')
    for j in range(predictions.shape[0]):
        print(str(i)+'_'+str(j))
        #Conv
        activations_EOG_dir2_conv = activations[1][0][j][na,:,na,:]
        weights_EOG_dir2_conv = model.CNNBlocks['eog'].dir2[0].weight.data.numpy()[:,:,na,:]
        bias_EOG_dir2_conv = model.CNNBlocks['eog'].dir2[0].bias.data.numpy()
        EOG_dir2_conv = Convolution_first_layer(activations_EOG_dir2_conv,weights_EOG_dir2_conv,bias_EOG_dir2_conv,filtersize=(1,500,2,64),stride=(1,1))
        R_EOG_dir2 = EOG_dir2_conv.lrp(R_EOG_dir2_save[j][na,:,:,:])

    for j in range(predictions.shape[0]):
        if predictions[j] == 0:
            R_wake[0] += np.sum(R_EEG_dir1[j,0,:,:])
            R_wake[0] += np.sum(R_EEG_dir2[:,0,:,:])
            R_wake[1] += np.sum(R_EEG_dir1[j,1,:,:])
            R_wake[1] += np.sum(R_EEG_dir2[:,1,:,:])
            R_wake[2] += np.sum(R_EOG_dir1[j,0,:,:])
            R_wake[2] += np.sum(R_EOG_dir2[:,0,:,:])
            R_wake[3] += np.sum(R_EOG_dir1[j,1,:,:])
            R_wake[3] += np.sum(R_EOG_dir2[:,1,:,:])
            R_wake[4] += np.sum(R_EMG[j])

            count1 += 1
        elif predictions[j] == 1:
            R_N1[0] += np.sum(R_EEG_dir1[j,0,:,:])
            R_N1[0] += np.sum(R_EEG_dir2[:,0,:,:])
            R_N1[1] += np.sum(R_EEG_dir1[j,1,:,:])
            R_N1[1] += np.sum(R_EEG_dir2[:,1,:,:])
            R_N1[2] += np.sum(R_EOG_dir1[j,0,:,:])
            R_N1[2] += np.sum(R_EOG_dir2[:,0,:,:])
            R_N1[3] += np.sum(R_EOG_dir1[j,1,:,:])
            R_N1[3] += np.sum(R_EOG_dir2[:,1,:,:])
            R_N1[4] += np.sum(R_EMG[j])
            
            count2 += 1
        elif predictions[j] == 2:
            R_N2[0] += np.sum(R_EEG_dir1[j,0,:,:])
            R_N2[0] += np.sum(R_EEG_dir2[:,0,:,:])
            R_N2[1] += np.sum(R_EEG_dir1[j,1,:,:])
            R_N2[1] += np.sum(R_EEG_dir2[:,1,:,:])
            R_N2[2] += np.sum(R_EOG_dir1[j,0,:,:])
            R_N2[2] += np.sum(R_EOG_dir2[:,0,:,:])
            R_N2[3] += np.sum(R_EOG_dir1[j,1,:,:])
            R_N2[3] += np.sum(R_EOG_dir2[:,1,:,:])
            R_N2[4] += np.sum(R_EMG[j])

            count3 += 1
        elif predictions[j] == 3:
            R_N3[0] += np.sum(R_EEG_dir1[j,0,:,:])
            R_N3[0] += np.sum(R_EEG_dir2[:,0,:,:])
            R_N3[1] += np.sum(R_EEG_dir1[j,1,:,:])
            R_N3[1] += np.sum(R_EEG_dir2[:,1,:,:])
            R_N3[2] += np.sum(R_EOG_dir1[j,0,:,:])
            R_N3[2] += np.sum(R_EOG_dir2[:,0,:,:])
            R_N3[3] += np.sum(R_EOG_dir1[j,1,:,:])
            R_N3[3] += np.sum(R_EOG_dir2[:,1,:,:])
            R_N3[4] += np.sum(R_EMG[j])

            count4 += 1
        else:
            R_REM[0] += np.sum(R_EEG_dir1[j,0,:,:])
            R_REM[0] += np.sum(R_EEG_dir2[:,0,:,:])
            R_REM[1] += np.sum(R_EEG_dir1[j,1,:,:])
            R_REM[1] += np.sum(R_EEG_dir2[:,1,:,:])
            R_REM[2] += np.sum(R_EOG_dir1[j,0,:,:])
            R_REM[2] += np.sum(R_EOG_dir2[:,0,:,:])
            R_REM[3] += np.sum(R_EOG_dir1[j,1,:,:])
            R_REM[3] += np.sum(R_EOG_dir2[:,1,:,:])
            R_REM[4] += np.sum(R_EMG[j])

            count5 += 1


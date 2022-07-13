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

model = load_model('/RQ2/LRP/WithoutSL/cnn_weightedloss.tar')


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
    print(i)
    predictions = np.load('/RQ2/LRP/WithoutSL/Predictions/prediction'+str(i)+'.npy',allow_pickle=True)
    activations1 = np.load('/RQ2/LRP/WithoutSL/Activations/activation'+str(i)+'_1.npy',allow_pickle=True)
    activations2 = np.load('/RQ2/LRP/WithoutSL/Activations/activation'+str(i)+'_2.npy',allow_pickle=True)
    activations1 = np.squeeze(activations1, axis=1)
    activations2 = np.squeeze(activations2, axis=1)
    activations2 = softmax(activations2,axis=1)
    #Linear Layer
    activations_linear = activations1
    activations_output = activations2
    weights_linear = model.linear.weight.data.numpy().T
    bias_linear = model.linear.bias.data.numpy()
    linear = Linear(6720,5,activations_linear,weights_linear,bias_linear)
    R = linear.lrp(activations_output)
    for j in range(predictions.shape[0]):
        if predictions[j] == 0:
            for k in range(5):
                R_wake[k] += np.sum(R[j][k*1344:(k+1)*1344])
            count1 += 1
        elif predictions[j] == 1:
            for k in range(5):
                R_N1[k] += np.sum(R[j][k*1344:(k+1)*1344])
            count2 += 1
        elif predictions[j] == 2:
            for k in range(5):
                R_N2[k] += np.sum(R[j][k*1344:(k+1)*1344])
            count3 += 1
        elif predictions[j] == 3:
            for k in range(5):
                R_N3[k] += np.sum(R[j][k*1344:(k+1)*1344])
            count4 += 1
        else:
            for k in range(5):
                R_REM[k] += np.sum(R[j][k*1344:(k+1)*1344])
            count5 += 1

R = []
R.append(R_wake/count1)
R.append(R_N1/count2)
R.append(R_N2/count3)
R.append(R_N3/count4)
R.append(R_REM/count5)

for j in range(0,5):
    R[j] = R[j]/np.sum(R[j])




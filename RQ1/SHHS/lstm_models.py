import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device, n_layers=2, drop_prob=0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_size,hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, bidirectional=True, batch_first=True)
        self.device = device

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers*2, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.n_layers*2, batch_size, self.hidden_size).to(self.device))
        
class Attention(nn.Module):
    def __init__(self, hidden_size, seq_length):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.fc1 = nn.Linear(hidden_size, 1, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, lstm_out, encoder_outputs):
        x1 = self.fc1(lstm_out)
        x2 = self.fc2(encoder_outputs)
        out = torch.tanh(x1+x2)
        return out.view(-1,self.seq_length)
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, seq_length, attention):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_length =seq_length
        self.attention = attention
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,num_layers=2, batch_first=True)
        self.rc = nn.Linear(4032, self.hidden_size*2)
        self.dropout2= nn.Dropout(0.5)
        self.linear = nn.Linear(self.hidden_size*4, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs, residual_connection_inputs):
        embedded = self.embedding(inputs)
        embedded = self.dropout1(embedded)
        lstm_out, hidden = self.lstm(embedded, hidden)
        alignment_scores = self.attention(lstm_out,encoder_outputs)
        attn_weights = F.softmax(alignment_scores, dim=1).view(-1,1,self.seq_length)
        context_vector = torch.bmm(attn_weights,encoder_outputs)
        combine_info = torch.cat((lstm_out, context_vector),-1)
        rc_info = self.rc(residual_connection_inputs)
        rc_info = self.dropout2(rc_info)
        combine_info = torch.cat((combine_info, rc_info),-1)
        output = self.linear(combine_info)
        return output, hidden, attn_weights
    
class SleepMultiChannelNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, device):
        super(SleepMultiChannelNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.encoder = Encoder(input_size, hidden_size, device)
        self.decoder = Decoder(hidden_size,output_size,seq_length,Attention(hidden_size, seq_length))

    def forward(self, decoder_input_first, decoder_input, cnn_features):
        cnn_features = cnn_features.view(-1,self.seq_length,4032)
        batch_size = cnn_features.shape[0]
        
        decoder_initialisation = decoder_input_first.view(1,1,-1)
        decoder_input = decoder_input.view(batch_size,self.seq_length,-1)
        for k in range(batch_size-1):
            decoder_initialisation = torch.cat((decoder_initialisation,decoder_input[k,self.seq_length-1,:].view(1,1,-1)),0)
        decoder_input_next = decoder_input[batch_size-1,self.seq_length-1,:].view(1,1,-1)
        
        encoder_initial_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(cnn_features, encoder_initial_hidden)

        encoder_hidden = (torch.cat((torch.mean(encoder_hidden[0][0:2], dim=0,keepdim=True),torch.mean(encoder_hidden[0][2:4], dim=0,keepdim=True)),0)
                         ,torch.cat((torch.mean(encoder_hidden[1][0:2], dim=0,keepdim=True),torch.mean(encoder_hidden[1][2:4], dim=0,keepdim=True)),0))
        encoder_outputs = torch.mean(encoder_outputs.view(batch_size,-1,2,self.hidden_size),dim=2,keepdim=False)
        
        decoder_inputs = decoder_initialisation
        decoder_hidden = encoder_hidden
        
        outputs = []
        for j in range(self.seq_length):
            decoder_outputs, decoder_hidden, attention_weights = self.decoder(decoder_inputs, decoder_hidden, encoder_outputs, cnn_features[:,j,:].view(batch_size,1,-1))
            decoder_inputs = decoder_outputs
            decoder_outputs_save = decoder_outputs.view(batch_size,-1)
            outputs.append(decoder_outputs_save)
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0,1)
        outputs = torch.reshape(outputs, (-1,5))
        return outputs, decoder_input_next








from torch.nn.modules.module import _addindent
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Sequence(nn.Module):
    def __init__(self,ninp,nh,nout,ipia):
        super(Sequence, self).__init__()
        self.lstm_pia = nn.LSTMCell(ninp, nh)
        self.linear_pia = nn.Linear(nh, nout)
        self.lstm1 = nn.LSTMCell(ninp+ipia, nh)
        self.lstm2 = nn.LSTMCell(nh, nh)
        self.linear = nn.Linear(nh, nout)
        self.nh=nh
        self.ninp=ninp
        self.nout=nout
        self.ipia=ipia
        print(ninp)
    #@torch.jit.script    
    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.nh, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.nh, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.nh, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.nh, dtype=torch.double) #(ns,self.nh)
        if self.ipia==1:
            h_t_pia = torch.zeros(input.size(0), self.nh, dtype=torch.double)
            c_t_pia = torch.zeros(input.size(0), self.nh, dtype=torch.double)
            output_pias=[]
            for input_t in input[:,:-1].split(self.ninp, dim=1):
                h_t_pia, c_t_pia = self.lstm_pia(input_t, (h_t_pia, c_t_pia)) 
                output_pia = self.linear(h_t_pia)
                output_pias+=[output_pia]
            output_pias=torch.cat(output_pias,dim=1)
            output_pias=torch.abs(output_pias)
            lstm_pia=output_pias.sum(axis=1)
            output_pias=torch.multiply(output_pias,input[:,-1,None])

        if self.ipia==1:
            for i,input_t in enumerate(input[:,:-1].split(self.ninp, dim=1)):
                input_t=torch.cat([input_t,output_pias[:,i:i+1]],dim=1)
                h_t, c_t = self.lstm1(input_t, (h_t, c_t)) #(ns,self.nh)
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
                outputs += [output]
            outputs+=[output_pia]
        else:
            for i,input_t in enumerate(input[:,:].split(self.ninp, dim=1)):
                h_t, c_t = self.lstm1(input_t, (h_t, c_t)) #(ns,self.nh)
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
                outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

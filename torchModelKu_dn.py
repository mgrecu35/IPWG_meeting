import matplotlib
import matplotlib.pyplot as plt
from torch.nn.modules.module import _addindent
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

                      
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

class Sequence(nn.Module):
    def __init__(self,ninp,nh):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(ninp, nh)
        self.lstm2 = nn.LSTMCell(nh, nh)
        self.linear = nn.Linear(nh,2)
        self.nh=nh
        self.ninp=ninp
        self.nout=1
        self.ipia=1
        print(ninp)
    #@torch.jit.script    
    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.nh, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.nh, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.nh, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.nh, dtype=torch.double) #(ns,self.nh)
        #print(input.shape)
        for i,input_t in enumerate(input[:,:].split(self.ninp, dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) #(ns,self.nh)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output[:,0:1]]
        outputs+=[output[:,1:2]]
        #print(outputs)
        #print(len(outputs))
        #outputs=torch.tensor(outputs)
        #print(outputs.shape)
        outputs = torch.cat(outputs, dim=1)
        return outputs

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, input_, target):
        self.input=input_
        self.target=target
        
    def __getitem__(self, index):
        # stuff
        return (self.input[index,:], self.target[index,:])

    def __len__(self):
        return self.input.shape[0]

    
from netCDF4 import Dataset
import glob


np.random.seed(0)
torch.manual_seed(0)
                  
#idf=0
#ipia=0
fh=Dataset("kuTrainingDataSetRand.nc")

pRate=fh["pRate"][:]
zKu=fh["zKu"][:]
piaKu=fh["piaKu"][:]
dn=fh["dn"][:]
input_=[]
target_=[]

for i,pRate1 in enumerate(pRate):
    i1=list(pRate1)
    i1.extend(dn[i,:])
    input_.append(i1)
    t1=list(zKu[i,:])
    t1.append(piaKu[i])
    target_.append(t1)

input_=np.array(input_)
target_=np.array(target_)

from sklearn.preprocessing import StandardScaler
import sys, argparse



stdScaler_input=StandardScaler()
stdScaler_target=StandardScaler()

nt=len(input_)

r=np.random.rand(nt)

stdScaler_input.fit(input_)
stdScaler_target.fit(target_)
a=np.nonzero(r<0.5)
b=np.nonzero(r>0.5)
input_n=stdScaler_input.transform(input_)
target_n=stdScaler_target.transform(target_)

d={'inputScaler': stdScaler_input, 'targetScaler': stdScaler_target}
import pickle
pickle.dump(d,open("scalers_Ku_forward.pklz","wb"))


t_input_n=torch.tensor(input_n[a[0],:])
t_target_n=torch.tensor(target_n[a[0],:])

v_input_n=torch.tensor(input_n[b[0],:])
v_target_n=torch.tensor(target_n[b[0],:])
                      
training_data=MyCustomDataset(t_input_n,t_target_n)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

np.random.seed(0)
torch.manual_seed(0)

lstm_model = Sequence(2,10)
lstm_model.double()

criterion = nn.MSELoss()

optimizer = optim.Adam(lstm_model.parameters())#,lr=0.5, momentum=0.1)#, lr=0.8)


for i in range(30):
    print('STEP: ', i)
    loss_av=[]
    for x,y in train_dataloader:
        def closure():
            optimizer.zero_grad()
            out = lstm_model(x)
            loss = criterion(out, y)
            loss_av.append(loss.item())
            #print('loss:', )
            loss.backward()
            return loss
        optimizer.step(closure)
    print(np.mean(loss_av))
    if i%3==0:
        torch.save(lstm_model.state_dict(),"lstm_forward_model_Ku_rev_rst.pt")
        
y_=lstm_model(v_input_n)
y_=y_[:,-2].detach().numpy()
target2_=v_target_n[:,-2].detach().numpy()
s=stdScaler_target.scale_[-2]
m=stdScaler_target.mean_[-2]
print(np.corrcoef(y_*s+m,target2_*s+m))
y1_=lstm_model(v_input_n)
y1_=y1_[:,-1].detach().numpy()
target1_=v_target_n[:,-1].detach().numpy()
s=stdScaler_target.scale_[-1]
m=stdScaler_target.mean_[-1]
print(np.corrcoef(y1_*s+m,target1_*s+m))


stop
plt.figure(figsize=(5,5))
ax=plt.subplot(111)
c=plt.hist2d(target2_*s+m,y_*s+m,bins=np.arange(60),cmap='jet',norm=matplotlib.colors.LogNorm())
ax.set_aspect('equal')
plt.xlim(0,60)
plt.ylim(0,60)
plt.title("Dual Frequency without SRT PIA")
plt.ylabel("Estimated rain rate (mm/h)")
plt.xlabel("True rain rate (mm/h)")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.15, 0.05, 0.7])
cb=plt.colorbar(cax=cax)
cb.ax.set_title('Counts')
plt.savefig("sfcRainHist2d_%2.2i_%2.2i.png"%(idf,ipia))

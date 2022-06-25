#import xarray as xr
import numpy as np
from netCDF4 import Dataset
import lidar
import matplotlib.pyplot as plt
fh=Dataset("lidarInput_3d.nc",'r')


iwc=fh["ls_radice"][:]
zku=fh["zKu"][:]
bscat=fh["pnorm3D"][:]
rho=fh['rho'][:]

#a=np.nonzero(iwc.sum(axis=0))
zkum=np.ma.array(zku,mask=zku<-10)
zku_mean=zkum.mean(axis=-1).mean(axis=-1)
nz,ny,nx=zku.shape
dn=fh['dn3d'][:]

a=np.nonzero(iwc.sum(axis=0)>1.5e-2)
yobsL=[]
xL=[]
for j,i in zip(a[0],a[1]):
     y1=np.log10(bscat[20:54,j,i]+1e-9)+5
     y1[y1<-1.5]=-1.5
     y1=list(y1)
     z1=zku[21:40,j,i]
     z1[z1<10]=0
     y1.extend(z1)
     yobsL.append(y1[::-1])
     xL.append(np.log10(iwc[21:40,j,i]*1e3*rho[21:40]))

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=30,weights='distance')
from sklearn.model_selection import train_test_split
yobsL=np.array(yobsL)
xL=np.array(xL)
X_train, X_test, y_train, y_test = train_test_split(yobsL[:,::1], xL[:,::1], test_size=0.33, random_state=42)
neigh.fit(X_train, y_train)
stop
         
plt.figure(figsize=(12,6))
plt.subplot(141)
for i in range(10):
    plt.plot(np.log10(bscat[:,i*10,100]),range(64))
    

plt.xlim(-7,-4)
plt.ylim(20,55)
plt.grid()

plt.subplot(142)
for i in range(10):
    plt.plot((zku[:,i*10,100]),range(64))

plt.xlim(-20,25)
plt.ylim(20,55)
plt.grid()

plt.subplot(143)
for i in range(10):
    plt.semilogx(1e3*(iwc[:,i*10,100]),range(64))

plt.xlim(0.1,4)
plt.ylim(20,55)
plt.grid()

plt.subplot(144)
for i in range(10):
    plt.plot((dn[:,i*10,100]),range(64),'*')

plt.xlim(-1,2)
plt.ylim(20,55)
plt.grid()

for i in range(20,54):
    a=np.nonzero(dn[i,:,:]>-99)
    print(dn[i,:,:][a].mean(axis=0))
stop
zku_mean=[]
zku_std=[]
bscatt_mean=[]
bscatt_std=[]
iwc_mean=[]
iwc_std=[]
for k in range(nz):
    a=np.nonzero(zku[k,:,:]>-10)
    #print(len(a[0]))
    if len(a[0])>0:
        zku_mean.append(zku[k,:,:][a].mean())
        zku_std.append(zku[k,:,:][a].std())
    else:
        zku_mean.append(-10)
        zku_std.append(0)
    b=np.nonzero(bscat[k,:,:]>1e-8)
    #print(len(a[0]))
    if len(b[0])>0:
        bscatt_mean.append(np.log10(bscat[k,:,:][b]).mean())
        bscatt_std.append(np.log10(bscat[k,:,:][b]).std())
    else:
        bscatt_mean.append(-7)
        bscatt_std.append(0)

    c=np.nonzero(iwc[k,:,:]>1e-4)
    if len(c[0])>0:
        iwc_mean.append(np.log10(iwc[k,:,:][a]).mean())
        iwc_std.append(np.log10(iwc[k,:,:][a]).std())
    else:
        iwc_mean.append(-4)
        iwc_std.append(0)
        
zku_mean=np.array(zku_mean)
zku_std=np.array(zku_std)
bscatt_mean=np.array(bscatt_mean)
bscatt_std=np.array(bscatt_std)

iwc_mean=np.array(iwc_mean)
iwc_std=np.array(iwc_std)
bscat[bscat<1e-8]=1e-8
a1=np.nonzero(iwc.sum(axis=0)>1.5e-2)
zkuL=[]
bscattL=[]
zku[zku<-10]=-10
iwcL=[]
for j1,i1 in zip(a1[0],a1[1]):
    zku1=(zku[:,j1,i1]-zku_mean)/(zku_std+1e-1)
    zku1[zku[:,j1,i1]<10]=-1.5
    zkuL.append(zku1[16:52][::-1])
    bscat1=(np.log10(bscat[:,j1,i1])-bscatt_mean)/(bscatt_std+1e-1)
    bscattL.append(bscat1[16:52][::-1])
    iwc1=(np.log10(iwc[:,j1,i1]+1e-5)-iwc_mean)/(iwc_std+1e-1)
    iwcL.append(iwc1[16:52][::-1])

import pickle
pickle.dump({"zku_n":np.array(zkuL),\
             "bscat_n":np.array(bscattL),"iwc_n":np.array(iwcL)},
            open("LRadarTraining.pklz","wb"))


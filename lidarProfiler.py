import numpy as np
from netCDF4 import Dataset
import lidar
import matplotlib.pyplot as plt
fh=Dataset("lidarInput_3d_99_08_27.nc",'r')


iwc=fh["ls_radice"][:]
zku=fh["zKu"][:]
bscat=fh["pnorm3D"][:]
betatot=fh["betatot3D"][:]
rho=fh['rho'][:]
Dm=fh['ls_radice'][:]
#range(10):
#for i in [3,0]:
for i in [3,0]:
    plt.figure()
    plt.suptitle("%i"%(i*10))
    plt.subplot(121)
    plt.plot(np.log10(betatot[:,i*10,100]),range(64))
    plt.plot(np.log10(bscat[:,i*10,100]),range(64))
    plt.ylim(17,50)
    plt.subplot(122)
    plt.plot(zku[:,i*10,100],range(64))
    plt.xlim(-20,20)
    plt.ylim(17,50)
    s1=(1-2*14.72*bscat[18:47,i*10,100].sum()*109)

#a=np.nonzero(iwc>1e-6)
a=np.nonzero(iwc.sum(axis=0)>0.75*1e-2)
dataL=[]
import tqdm
for iv in tqdm.tqdm(range(a[0].shape[0])):
    j, i=a[0][iv], a[1][iv]#in zip(a[0],a[1]):
    a1=np.nonzero(bscat[:,j,i]>1e-6)
    bscat1=bscat[:,j,i][a1][::-1].cumsum()
    for ik,k in enumerate(a1[0][::-1]):
        data1=[np.log10(bscat[k,j,i]),\
               np.log10(bscat1[ik]),zku[k,j,i],50-k,iwc[k,j,i]*rho[k]*1e3]
        #print(data1)
        dataL.append(data1)
    #stop

import xarray as xr

d=xr.Dataset({"xy_data":xr.DataArray(dataL)})

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in d.data_vars}
#ds.to_netcdf(filename)
d.to_netcdf("trainingDataset_99_08_27.nc", encoding=encoding)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib

dataL=np.array(dataL)
neigh = [KNeighborsRegressor(n_neighbors=30,weights='distance') for i in range(3)]
X_train, X_test, \
    y_train, y_test = train_test_split(dataL[:,:4], dataL[:,-1], \
                                       test_size=0.33, random_state=42)

neigh[0].fit(X_train[:,[0,1,3]], y_train)
y1_=neigh[0].predict(X_test[:,[0,1,3]])

neigh[1].fit(X_train[:,[0,1]], y_train)
y2_=neigh[1].predict(X_test[:,[0,1]])


fig=plt.figure(figsize=(12,7))
ax1=plt.subplot(121)
c1=plt.hist2d(y1_,y_test,bins=10**(-2.5+np.arange(30)*0.1),cmap='jet',norm=matplotlib.colors.LogNorm())
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.01,1)
plt.ylim(0.01,1)
#ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
#ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.grid()
#ax.axhline(0.5,color='gray')
ax1.set_aspect('equal')
plt.xlabel('True IWC [g/m^3]')
plt.ylabel('Estimated IWC [g/m^3]')
plt.title('With range information')
#plt.colorbar()
#cbar=plt.colorbar(c1,orientation='horizontal')
#cbar.ax.set_title('Counts')

ax2=plt.subplot(122)
c2=plt.hist2d(y2_,y_test,bins=10**(-2.5+np.arange(30)*0.1),cmap='jet',norm=matplotlib.colors.LogNorm())
plt.yscale('log')
plt.xscale('log')
#ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
#ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.xlim(0.01,1)
plt.ylim(0.01,1)
plt.grid()
#ax.axhline(0.5,color='gray')
ax2.set_aspect('equal')
plt.xlabel('True IWC [g/m^3]')
plt.ylabel('Estimated IWC [g/m^3]')
plt.title('Without range information')


fig.subplots_adjust(bottom=0.25)
l1, b1, w1, h1= ax1.get_position().bounds
cbar_ax = fig.add_axes([l1, 0.05, w1, 0.05])
cb1=fig.colorbar(c1[-1], cax=cbar_ax, orientation='horizontal')
cb1.ax.set_title('Counts')
l2, b2, w2, h2= ax2.get_position().bounds
cbar_ax2 = fig.add_axes([l2, 0.05, w1, 0.05])
cb2=fig.colorbar(c2[-1], cax=cbar_ax2, orientation='horizontal')
cb2.ax.set_title('Counts')
plt.savefig('lidarOnly.png')



dataL=np.array(dataL)
a=np.nonzero(dataL[:,2]>12)
dataLZ=dataL[a[0],:]
neigh_Z = [KNeighborsRegressor(n_neighbors=30,weights='distance') for i in range(3)]
Xz_train, Xz_test, \
    yz_train, yz_test = train_test_split(dataLZ[:,:4], dataLZ[:,-1], \
                                       test_size=0.33, random_state=42)

neigh_Z[0].fit(Xz_train[:,[0,1,2,3]], yz_train)
y1z_=neigh_Z[0].predict(Xz_test[:,[0,1,2,3]])

neigh_Z[1].fit(Xz_train[:,[0,1,2]], yz_train)
y2z_=neigh_Z[1].predict(Xz_test[:,[0,1,2]])

from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter

fig=plt.figure(figsize=(12,7))
ax1=plt.subplot(121)
c1=plt.hist2d(yz_test,y1z_,bins=10**(-2.0+np.arange(40)*0.05),cmap='jet',norm=matplotlib.colors.LogNorm())
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.1,1)
plt.ylim(0.1,1)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
ax1.xaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.minorticks_off()
plt.grid()
ax1.set_aspect('equal')
plt.xlabel('True IWC [g/m^3]')
plt.ylabel('Estimated IWC [g/m^3]')
plt.title('With range information')

ax2=plt.subplot(122)
c2=plt.hist2d(yz_test,y2z_,bins=10**(-2.0+np.arange(40)*0.05),cmap='jet',norm=matplotlib.colors.LogNorm())
plt.yscale('log')
plt.xscale('log')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
ax2.xaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
ax2.minorticks_off()
plt.xlim(0.1,1)
plt.ylim(0.1,1)
plt.grid()
#ax.axhline(0.5,color='gray')
ax2.set_aspect('equal')
plt.xlabel('True IWC [g/m^3]')
plt.ylabel('Estimated IWC [g/m^3]')
plt.title('Without range information')


fig.subplots_adjust(bottom=0.25)
l1, b1, w1, h1= ax1.get_position().bounds
cbar_ax = fig.add_axes([l1, 0.05, w1, 0.05])
cb1=fig.colorbar(c1[-1], cax=cbar_ax, orientation='horizontal')
cb1.ax.set_title('Counts')
l2, b2, w2, h2= ax2.get_position().bounds
cbar_ax2 = fig.add_axes([l2, 0.05, w1, 0.05])
cb2=fig.colorbar(c2[-1], cax=cbar_ax2, orientation='horizontal')
cb2.ax.set_title('Counts')
plt.savefig('combinedRadarLidar.png')

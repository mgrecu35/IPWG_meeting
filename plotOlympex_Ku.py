from readLinData import *
from   numpy import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator,LogFormatter
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LogNorm
import pickle


n1=3000
n2=5600
fname='RADEX_HIWRAP_L1B_20151201_225755-20151201_235943_HKu_dist_v01a.nc'
fname="olympex_CRS_20151201_225847-20151201_235952_2_v01a.nc"
fname="RADEX_HIWRAP_L1B_20151123_165702-20151123_180235_HKu_dist_v01a.nc"
fname="olympex_CRS_20151123_165814-20151123_175918_2_v01a.nc"
zku,vku,lat,lon,r,alt,roll,t=readiphexKu(fname)
zkum=ma.array(zku,mask=zku<-20)
hm=alt[n1:n2].mean()
rmean=r
plt.pcolormesh(t[n1:n2:5],(hm-rmean)/1e3,zkum[n1:n2:5,:].T,vmin=-20,vmax=30,cmap='jet')
plt.ylim(0,10)
plt.colorbar()
#stop
"olympex_radex_CPL_ATB_16921_20151201.hdf5"
lidar="olympex_radex_CPL_ATB_16919_20151123.hdf5"
fhl=Dataset(lidar)
t_lidar=fhl["Hour"][:]+fhl["Minute"][:]/60+fhl["Second"][:]/3600
import numpy as np
beta_532=fhl["ATB_1064"][:]
beta_1064=fhl["ATB_1064"][:]
hl=fhl["Bin_Alt"][:]
plt.figure()
a=np.nonzero((t_lidar-t[n1])*(t_lidar-t[n2])<0)
beta_532_m=np.ma.array(beta_532,mask=beta_532<1e-6)
from scipy.ndimage import gaussian_filter
beta_532_m=gaussian_filter(beta_532_m,sigma=1)
beta_532_m=np.ma.array(beta_532_m,mask=beta_532_m<1e-6)
c=plt.pcolormesh(t_lidar[a],hl,beta_532_m[a[0],:].T,\
                 norm=matplotlib.colors.LogNorm(),\
                 cmap='jet')
zTop=[]
zBot=[]
for i in a[0]:
    a1=np.nonzero(beta_532_m[i,320:600].data>3e-4)
    zTop.append(hl[320:600][a1[0][0]])
    a2=np.nonzero(beta_532_m[i,320:600].data>5e-4)
    zBot.append(hl[320:600][a2[0][-1]])

zBot=gaussian_filter(zBot,sigma=5)
zTop=gaussian_filter(zTop,sigma=5)
dh=hl[0]-hl[1]
h1=((hm-rmean)/1e3)[::-1]
bscattL=[]
zwL=[]
ikL=[]
ikbL=[]
for ip,i in enumerate(a[0]):
    ik1=int(-(zTop[ip]-hl[0])/dh)
    ik2=int(-(zBot[ip]-hl[0])/dh)
    backscatt=beta_532_m[i,ik1:ik2]
    ind_r=np.argmin(np.abs(t_lidar[i]-t[n1:n2]))
    zw_int=np.interp(hl[ik1:ik2][::-1],h1,zku[ind_r+n1,::-1])
    ikL.append(ik1)
    ikbL.append(ik2)
    bscattL.append(backscatt)
    zwL.append(zw_int[::-1])
    print(ik1,ik2)

import pickle
pickle.dump({"bscatt":bscattL,"zw":zwL,"dh":dh,"ikL":ikL},open("olympex_Nov23.pklz","wb"))
stop
plt.plot(t_lidar[a],zTop)
plt.plot(t_lidar[a],zBot)
plt.colorbar(c)
#plt.contour(t[n1:n2],(hm-rmean)/1e3,zkum[n1:n2,:].T,levels=[-20,-15],\
#            colors=['black' for k in range(2)])
plt.ylim(4,10)
plt.figure()
#plt.plot(beta_532_m[a[0][100]],hl)
#plt.plot(beta_1064[a[0][100]],hl)
#plt.ylim(8,12)
#stop

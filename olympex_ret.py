import pickle
d=pickle.load(open("olympex_Nov23.pklz","rb"))
bscatt=d["bscatt"]
zw=d["zw"]
dh=d["dh"]
from netCDF4 import Dataset
fh=Dataset("trainingDataset_99_08_27_w_lidar.nc")
xydata=fh['xy_data'][:]
ikL=d["ikL"]
# 1-log10(bscatt), 2 -- log10(integral(bscatt)), 3 -- zw

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib
import numpy as np

neigh = KNeighborsRegressor(n_neighbors=50,weights='distance')
a1=np.nonzero(xydata.data[:,0]>-7)
neigh.fit(xydata.data[a1[0],:3], xydata[a1[0],-1])


for i,b1 in enumerate(bscatt):
    iwcRet1=np.zeros((181),float)
    zw1=zw[i]
    xL=[]
    kL=[]
    for k, b11 in enumerate(b1):
        if zw1[k]>-20 and b11>0:
            x1=[np.log10(b11)-3,np.log10(b1[0:k+1].sum()*dh),zw1[k]]
            xL.append(x1)
            kL.append(k)

    stop

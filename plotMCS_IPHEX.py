n1=0
n2=900
#n1=1700
#n2=1900
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
files=glob.glob("Data/*HDF5")
from scipy.ndimage import gaussian_filter
import numpy as np
orbits=[46472,46481,46488,46718,46795,46826,46872,46503,46579,\
        46590,46605,46610,46620,46626,46666]
nprofs=0
bsfcL=[]
cfad=np.zeros((60,50),float)
cfadZKu=np.zeros((60,50),float)
cfadZKa=np.zeros((60,50),float)
pmean=np.zeros((60),float)
cmean=np.zeros((60),float)
zmean=np.zeros((60,2),float)
zkuL=[]
zkaL=[]
for fname in sorted(files):
    fh=Dataset(fname,'r')
    if int(fname.split('.')[-3]) not in orbits:
        continue
    zKu=fh['FS/PRE/zFactorMeasured'][n1:n2,:,:,0]
    zKa=fh['FS/PRE/zFactorMeasured'][n1:n2,:,:,1]
    bzd=fh['FS/VER/binZeroDeg'][n1:n2,:,]
    pType=(fh['FS/CSF/typePrecip'][n1:n2,:,]/1e7).astype(int)
    stormTop=(fh['FS/PRE/binStormTop'][n1:n2,:,]/1e7).astype(int)
    bcf=fh['FS/PRE/binClutterFreeBottom'][n1:n2,:,]
    s0=fh['FS/PRE/sigmaZeroMeasured'][n1:n2,:]
    s0Ka=fh['HS/PRE/sigmaZeroMeasured'][n1:n2,:]
    pia=fh['FS/SRT/pathAtten'][n1:n2,:]
    srtPIA=fh['FS/SRT/pathAtten'][:,:]
    relPIA=fh['FS/SRT/reliabFlag'][:,:]
    lon=fh["FS/Longitude"][n1:n2,:]
    lat=fh["FS/Latitude"][n1:n2,:]
    relFlag=fh['FS/SRT/reliabFlag'][n1:n2,:]
    sfcRain=fh['FS/SLV/precipRateNearSurface'][n1:n2,:]
    lon=fh['FS/Longitude'][:]
    lat=fh['FS/Latitude'][:]
    h0=fh['FS/VER/heightZeroDeg'][:]
    a=np.nonzero(h0[:,12:37]>3500)
    b=np.nonzero(pType[:,12:37][a]==2)
    bzd=fh['FS/VER/binZeroDeg'][:]
    bcf=fh['FS/PRE/binClutterFreeBottom'][:]
    bsfc=fh['FS/PRE/binRealSurface'][:]
    bsfcL.extend(bsfc[:,12:37,0][a][b])
    precipRate=fh['FS/SLV/precipRate'][:]
    #precipRateL.extend()
    bzdn=bzd[:,12:37]
    bcfn=bcf[:,12:37]
    precipRaten=precipRate[:,12:37,:]
    for i,precip1d in enumerate(zKu[:,12:37,:][a][b]):
        i1=a[0][b][i]
        j1=a[1][b][i]
        bzd1=bzdn[i1,j1]
        if zKu[i1,j1,0:bzd1-4].max()<22:
            continue
        zm_ku=zKu[i1,j1,bzd1-40:bzd1+20]
        zm_ka=zKa[i1,j1,bzd1-40:bzd1+20]
        prate1=precipRaten[i1,j1,bzd1-40:bzd1+20]
        ar=np.nonzero(prate1<1e-3)
        zm_ku[ar]=0
        zm_ku[zm_ku<0]=0
        zm_ka[ar]=0
        zm_ka[zm_ka<0]=0
        #print(zm_ku[ar])
        for k in range(bzd1-40,bzd1+20):
            if k<bcfn[i1,j1]:
                i0=int(precip1d[k]/2)
                if precipRaten[i1,j1,k]<=0.001:
                    continue
                i0=int(np.log10(precipRaten[i1,j1,k]/0.1)*15)
                cmean[k-bzd1+40]+=1
                pmean[k-bzd1+40]+=precipRaten[i1,j1,k]
                zmean[k-bzd1+40,0]+=max(0,zKu[i1,j1,k])
                zmean[k-bzd1+40,1]+=max(0,zKa[i1,j1,k])
                if i0>=0 and i0<50 and precip1d[k]>12:
                    cfad[k-bzd1+40,i0]+=1
                i0=int(zKu[i1,j1,k])
                if i0>=0 and i0<50 and precip1d[k]>12 and precipRate[i1,j1,k]>0:
                    cfadZKu[k-bzd1+40,i0]+=1
                i0=int(zKa[i1,j1,k])
                if i0>=0 and i0<50 and precip1d[k]>12 and precipRate[i1,j1,k]>0:
                    cfadZKa[k-bzd1+40,i0]+=1
        nprofs+=1
        zkuL.append(zm_ku)
        zkaL.append(zm_ka)
    print(sfcRain.sum())
    continue
    plt.figure(figsize=(8,12))
    plt.subplot(311)
    nx,ny=lat.shape
    sfcRain1d=sfcRain.sum(axis=-1)
    sfcRain1d=gaussian_filter(sfcRain1d,sigma=2)
    ind=np.argmax(sfcRain1d)
    i1=max(0,ind-40)
    i2=min(nx,ind+40)
    plt.pcolormesh(lon,lat,sfcRain,norm=matplotlib.colors.LogNorm(),cmap='jet')

    #plt.subplot(311)
    precipRatem=np.ma.array(precipRate, mask=precipRate<0.01)
    plt.pcolormesh(range(nx),np.arange(176)*0.125,
                   precipRatem[:,24,::-1].T,cmap='jet',norm=matplotlib.colors.LogNorm())
    plt.xlim(i1,i2)
    plt.ylim(0,12)
    plt.colorbar()
    zKum=np.ma.array(zKu,mask=zKu<0)
    zKam=np.ma.array(zKa,mask=zKa<0)
    plt.title('Orbit %i'%(int(fname[-16:-10])))
    plt.subplot(312)
    plt.pcolormesh(range(nx),np.arange(176)*0.125,zKum[:,24,::-1].T,cmap='jet',vmax=50)
    plt.xlim(i1,i2)
    plt.ylim(0,12)
    plt.colorbar()
    plt.subplot(313)
    plt.pcolormesh(range(nx),np.arange(176)*0.125,zKam[:,24,::-1].T,cmap='jet',vmax=50)
    plt.xlim(i1,i2)
    plt.ylim(0,12)
    plt.colorbar()
    plt.show()

import pickle
pickle.dump({"zku":np.array(zkuL),"zka":np.array(zkaL)},open("zprofs.pklz","wb"))

matplotlib.rcParams['font.size']=12
rbin=np.arange(50)/15
rr=10**rbin*0.1
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.pcolormesh(rr,(40-np.arange(60))*0.125,cfad,norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.xscale('log')
plt.xlabel('mm/h')
plt.ylabel('Relative height(km)')
plt.title('Precip. distribution')
cbar=plt.colorbar()
cbar.ax.set_title('Counts')
plt.subplot(122)
plt.plot(pmean/cmean,(40-np.arange(60))*0.125)
plt.ylabel('Relative height(km)')
plt.xlabel('mm/h')
plt.title('Mean precip. profile')
plt.savefig('convPrecip_2.png')

plt.figure(figsize=(12,8))
plt.subplot(121)
plt.pcolormesh(range(50),(40-np.arange(60))*0.125,cfadZKu,norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.xlim(5,50)
plt.title('Ku-band')
plt.xlabel('dBZ')
plt.ylabel('Relative Height (km)')
cbar=plt.colorbar()
cbar.ax.set_title('Count')

plt.subplot(122)
plt.pcolormesh(range(50),(40-np.arange(60))*0.125,cfadZKa,norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.xlim(5,35)
plt.title('Ka-band')
plt.xlabel('dBZ')
plt.ylabel('Relative Height (km)')
cbar=plt.colorbar()
cbar.ax.set_title('Count')
plt.savefig('convZCfads_2.png')
stop
zKum=np.ma.array(zKu,mask=zKu<0)

plt.subplot(211)
plt.pcolormesh(zKum[:,30,:].T,cmap='jet',vmax=50)
plt.ylim(175,60)
plt.xlim(25,45)
#plt.xlim(145,160)
plt.colorbar()
plt.subplot(212)
plt.pcolormesh(zKam[:,30,:].T,cmap='jet',vmax=35)
plt.xlim(25,45)
#plt.xlim(145,160)
plt.ylim(175,60)
plt.colorbar()
plt.figure()
plt.plot(lon[:,0],lat[:,0])
plt.plot(lon[:,-1],lat[:,-1])
import pickle
dloc=pickle.load(open("../scatter-1.1/zKuX_20140523.pklz","rb"))
plt.plot(dloc["lon"],dloc["lat"])
hlon=dloc["lon"][::1]
hlat=dloc["lat"][::1]
zKuLm=[]
for x,y in zip(hlon,hlat):
    rms=(x-lon)**2+(y-lat)**2
    ind=np.argmin(rms)
    i0=int(ind/49)
    j0=ind-i0*49
    j2=min(49,j0+2)
    j1=j0-1
    z1L=[]
    rmsL=[]
    ijL=[]
    for i in range(i0-1,i0+2):
        for j in range(j1,j2):
            z1L.append(zKum[i,j,:])
            rmsL.append(((x-lon[i,j])**2+(y-lat[i,j])**2)**0.5)
            ijL.append([i,j])
    ind=np.argsort(rmsL)
    zint=np.zeros((176),float)
    sumw=0
    for i in ind[0:1]:
        w=np.exp(-2.5*(rmsL[i]/0.01)**2)
        w=1./(rmsL[i]+0.0001)
        zint+=w*10**(0.1*z1L[i].data)
        sumw+=w
    zint=np.log10(zint/sumw)*10.0
    #stop
    #print(i0,j0)
    #
    #zKuLm.append(zKum[ijL[ind[0]][0],ijL[ind[0]][1],:])
    zKuLm.append(zKum[i0,j0,:])
    #zKuLm.append(zint)
    
#stop
#stop
zKuLm=np.array(zKuLm)
zKuLm=np.ma.array(zKuLm,mask=zKuLm<10)
plt.figure()
plt.subplot(111)
plt.pcolormesh(dloc["time"],np.arange(176)*0.125,zKuLm[:,::-1].T,cmap='jet',vmin=0,vmax=50)
plt.ylim(0,15)
plt.ylabel("Range")
plt.xlabel("Time")
plt.title("DPR Ku-band")
plt.colorbar()
plt.savefig('23May2014_DPR.png')
stop
#stop
pRate=fh['FS/SLV/precipRate'][n1:n2,:]
zKu[zKu<0]=0
zmL2=[]
pRateL=[]
piaL=[]
relFlagL=[]
sfcRainL=[]
for i1 in range(zKu.shape[0]):
    for j1 in range(20,28):
        if bzd[i1,j1]>stormTop[i1,j1]+4 and bcf[i1,j1]-bzd[i1,j1]>20 and\
           pType[i1,j1]==2:
            if bzd[i1,j1]-60>0 and bzd[i1,j1]+20<176:
                zmL2.append(zKu[i1,j1,bzd[i1,j1]-60:bzd[i1,j1]+20])
                pRateL.append(pRate[i1,j1,bzd[i1,j1]-60:bzd[i1,j1]+30])
                piaL.append(pia[i1,j1])
                relFlagL.append(relFlag[i1,j1])
                sfcRainL.append(sfcRain[i1,j1])
import pickle
som=pickle.load(open("miniSOM_Land.pklz","rb"))

nx=len(zmL2)
iclassL=[]
for it in range(nx):
    win=som.winner(zmL2[it])
    iclass=win[0]*3+win[1]+1
    iclassL.append(iclass)

plt.figure()
plt.plot(np.array(pRateL).mean(axis=0),-60+np.arange(90))
plt.ylim(30,-60)
plt.xlim(0,40)

plt.figure()

for i in range(nx):
    if iclassL[i]==9:
        plt.plot(zmL2[i],-60+np.arange(80))
plt.ylim(20,-60)

from sklearn.cluster import KMeans

plt.figure(figsize=(12, 12))

iclassL=np.array(iclassL)
a=np.nonzero(iclassL==9)
# Incorrect number of clusters
zmL2=np.array(zmL2)

kmeans = KMeans(n_clusters=16, random_state=10).fit(zmL2[a[0],:])
plt.figure()
zmAvg=[]
piaL=np.array(piaL)
sfcRainL=np.array(sfcRainL)
for i in range(16):
    a1=np.nonzero(kmeans.labels_==i)
    #plt.figure()
    zm1=zmL2[a[0][a1],:].mean(axis=0)
    if zm1.max()>47:
        plt.plot(zm1,-60+np.arange(80))
    zmAvg.append(zmL2[a[0][a1],:].mean(axis=0))
    plt.ylim(20,-60)
    #plt.title("PIA=%6.2f %6.2f"%(piaL[a[0][a1]].mean(),sfcRainL[a[0][a1]].mean()))
plt.xlabel('dBZ')
plt.ylabel('Relative range')
plt.savefig('deepConvProfs.png')
pickle.dump({"zmAvg":zmAvg},open("zmAvg.pklz","wb"))



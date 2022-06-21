import combAlg as cAlg
import pickle
import numpy as np
d=pickle.load(open("zprofs.pklz","rb"))
zka=d["zka"]
zku=d["zku"]
envParams=np.loadtxt("env.txt")
from netCDF4 import Dataset

fh=Dataset("modConvectiveDbase_Torch.nc")
zku=fh["zKu"][:,44:]
zka=fh["zKa"][:,44:]
wv=envParams[16:76,2]
temp=envParams[16:76,1]
press=envParams[16:76,2]

from netCDF4 import Dataset
fh=Dataset("scatteringTablesGPM_SPH.nc")
ng=272
zKuG=fh["zKuG"][0:272]
zKaG=fh["zKaG"][0:272]
attKaG=fh["attKaG"][0:272]
attKuG=fh["attKuG"][0:272]
dmG=fh["dmg"][:272]
gwc=fh["gwc"][:272]
graupRate=fh["graupRate"][:272]
zKuR=fh["zKuR"][0:272]
zKaR=fh["zKaR"][0:272]
attKaR=fh["attKaR"][0:272]
attKuR=fh["attKuR"][0:272]
dmR=fh["dmr"][:272]
rwc=fh["rwc"][:272]
rainRate=fh["rainRate"][:272]
ns=253
zKuS=fh["zKuS"][0:ns]
zKaS=fh["zKaS"][0:ns]
attKaS=fh["attKaS"][0:ns]
attKuS=fh["attKuS"][0:ns]
dmS=fh["dms"][:ns]
swc=fh["swc"][:ns]
snowRate=fh["snowRate"][:ns]

zka_predL=[[] for k in range(40)]
zka_trueL=[[] for k in range(40)]
pRate1L=[[] for k in range(70)]
dng=1-0.6*np.arange(70)/39
piaKaL=[]
piaKuL=[]
gRateL=[]
pRateL=[]
for i,zku1 in enumerate(zku):
    piaKa=0
    piaKu=0
    for k,zm in enumerate(zku1[:40]):
        if zm>10 and zka[i,k]>10:
            ibin=int((zm-10*dng[k]+12)/0.25)
            if ibin>ng-1:
                ibin=ng-1
            piaKa+=10**dng[k]*attKaG[ibin]*0.125
            piaKu+=10**dng[k]*attKuG[ibin]*0.125
            piaKu+=10**dng[k]*attKuG[ibin]*0.125
            zka_predL[k].append(zKaG[ibin]+10*dng[k]-piaKa)
            piaKa+=10**dng[k]*attKaG[ibin]*0.125
            zka_trueL[k].append(zka[i,k])
            prate1=graupRate[ibin]*10**dng[k]
            #prate1=0.28*10**(0.1*zm*0.46)#snowRate[ibin]*10**dng[k]
            pRate1L[k].append(prate1)
        else:
            pRate1L[k].append(0)
    for k,zm in enumerate(zku1[40:]):
        if zm>10:
            ibin=int((zm+piaKu-10*dng[k+40]+12)/0.25)
            if ibin>271:
                ibin=271
            prate2=rainRate[ibin]*10**dng[k+40]
            piaKu+=10**dng[k+40]*attKuR[ibin]*0.125*2
            pRate1L[k+40].append(prate2)
        else:
            pRate1L[k+40].append(0.)
    try:
        gRateL.append(prate1)
        piaKaL.append(piaKa)
        piaKuL.append(piaKu)
        ibin=int((zku1[44]-10*dng[39]+12)/0.25)
        prate2=rainRate[ibin]*10**dng[39]
        pRateL.append(prate2)
    except:
        continue
    
zKa_prof=[]

import matplotlib.pyplot as plt

for i in range(40): 
    zKa_prof.append([np.mean(zka_predL[i]),np.mean(zka_trueL[i])])

zKa_prof=np.array(zKa_prof)
plt.plot(zKa_prof[:,0],40*0.125-np.arange(40)*0.125)
plt.plot(zKa_prof[:,1],40*0.125-np.arange(40)*0.125)

plt.figure()
for i in range(40):
    plt.scatter(zka_trueL[k],zka_predL[k])

plt.figure()
plt.plot([np.mean(pRate1L[k]) for k in range(0,70)],70*0.125-np.arange
         (70)*0.125)

import numpy as np
pRate2D=np.array(pRate1L).T
import xarray as xr

pRate2Dx=xr.DataArray(pRate2D)
zKu_r=xr.DataArray(zku)
zKa_r=xr.DataArray(zka)
d=xr.Dataset({"pRate":pRate2Dx,"zKu":zKu_r,"zKa":zKa_r})
d.to_netcdf("convProfs.nc")

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
stdScaler = StandardScaler()
pRate2D_sc=stdScaler.fit_transform(pRate2D)
pca = PCA()
pca.fit(pRate2D_sc)
plt.figure()
plt.plot(pca.explained_variance_ratio_.cumsum()[:6])

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
zKuR=fh["zKuR"][0:289]
zKaR=fh["zKaR"][0:289]
attKaR=fh["attKaR"][0:289]
attKuR=fh["attKuR"][0:289]
dmR=fh["dmr"][:289]
rwc=fh["rwc"][:289]
rainRate=fh["rainRate"][:289]
ns=253
zKuS=fh["zKuS"][0:ns]
zKaS=fh["zKaS"][0:ns]
attKaS=fh["attKaS"][0:ns]
attKuS=fh["attKuS"][0:ns]
dmS=fh["dms"][:ns]
swc=fh["swc"][:ns]
snowRate=fh["snowRate"][:ns]



fh=Dataset("convProfs.nc")

pRate=fh["pRate"][:]
zKu_real=fh["zKu"][:]

dng=1-0.6*np.arange(70)/39

zKuL=[]
piaKuL=[]
from forwardModelSub import *
dnL=[]
from scipy.ndimage import gaussian_filter
for pRate1 in pRate:
    piaKa=0
    piaKu=0
    zKu1=[]
    for it in range(5):
        dng=1-0.6*np.arange(70)/39
        dng+=np.random.randn()*0.75
        ddn1=gaussian_filter(np.random.randn(70),sigma=5)
        dng+=ddn1
        zKu1,piaKu,piaKa=computeZKu(pRate1,dng,graupRate,attKaG,attKuG,\
                                    rainRate,attKaR,attKuR,\
                                    zKuG,zKuR,cAlg)        
    zKuL.append(zKu1)
    piaKuL.append(piaKu)
    dnL.append(dng)
zKuL=np.array(zKuL)
import xarray as xr

pRate2Dx=xr.DataArray(pRate)
zKuLx=xr.DataArray(zKuL)
piaKuLx=xr.DataArray(piaKuL)
dnLx=xr.DataArray(dnL)
d=xr.Dataset({"pRate":pRate2Dx,"zKu":zKuLx,"piaKu":piaKuLx,"dn":dnLx})
d.to_netcdf("kuTrainingDataSetRand.nc")


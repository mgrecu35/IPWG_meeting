from numpy import *

import glob
fileList=glob.glob("PSDs/IMPACTS_MergedH*")
wisperF=glob.glob("PSDs/IMPACTS-WISPER*nc")

wisperF=sorted(wisperF)
from netCDF4 import Dataset
fh=Dataset(fileList[0])

dbins=fh['CONCENTRATION'].bin_midpoints
dint=fh['CONCENTRATION'].bin_endpoints

dbins=array(dbins)
dint=array(dint)
ds=dint[1:]-dint[0:-1]
import numpy as np

iwcL1=[]
iwcL2=[]
tempCL=[]

dbins=np.array([   30. ,    50. ,    70. ,    90. ,   112.5,   137.5,   175. ,
                   225. ,   275. ,   325. ,   375. ,   437.5,   512.5,   587.5,
                   662.5,   750. ,   850. ,   950. ,  1100. ,  1300. ,  1500. ,
                   1700. ,  2000. ,  2400. ,  2800. ,  3200. ,  3600. ,  4000. ,
                   4400. ,  4800. ,  5500. ,  6500. ,  7500. ,  8500. ,  9500. ,
                   11000. , 13000. , 15000. , 17000. , 19000. , 22500. , 27500. ])

dint=fh['CONCENTRATION'].bin_endpoints
ds=dint[1:]-dint[:-1]
mD_Coeff=np.zeros((2),float)
mD_Coeff[0]=0.0061
mD_Coeff[1]=2.05


massB=[mD_Coeff[0]*(1e-4*dbins)**(-0.2+i*0.05+mD_Coeff[1]) for i in range(9)]

iwcL3=[]
iwcL1=[]
iwcL2=[]

import pytmatrix.refractive
wl=[pytmatrix.refractive.wl_Ku,pytmatrix.refractive.wl_Ka,pytmatrix.refractive.wl_W]
from scattering import *
from integrate import *
itemp=-3
ifract=14
massP=massB[4]*1e-3
msphere=np.pi*(dbins*1e-6)**3/6*980
fice=massP/msphere

bscatKu,qscatKu,qextKu,gKu,vfallKu=interp_mass(scatTables,'13.8',massB[4]*1e-3,itemp,ifract,fice)
bscatKa,qscatKa,qextKa,gKa,vfallKa=interp_mass(scatTables,'35',massB[4]*1e-3,itemp,ifract,fice)
bscatW,qscatW,qextW,gW,vfallW=interp_mass(scatTables,'94',massB[4]*1e-3,itemp,ifract,fice)
fscale=np.exp(-0.1*1e-3*dbins)
bscatKu*=fscale
bscatKa*=fscale
bscatW*=fscale

#stop
rhow=1e3 # kg/m^3
dat1L=[]
dbins_eq=(massB[4]/rhow/1e3*6/np.pi)**(1/3.)
#stop
for i,fname in enumerate(sorted(fileList[:])[:]):
    #print(fname,wisperF[i])
    fh=Dataset(fname)
    print(fname,wisperF[i])
    c=fh['CONCENTRATION'][:,:].T
    #stop
    iwc_ncar=fh['IWC'][:]
    t_ncar=fh['time'][:]
    a=np.nonzero(iwc_ncar>0)
    fhw=Dataset(wisperF[i])
    iwc_wsp=fhw["iwc_wisper"][:]
    tempC=fhw["tempC"][:]
    
    for i,Nbins in enumerate(c):
        if Nbins.min()<0 or Nbins.sum()<1e-3:
            continue
        if tempC[i]<-998:
            continue
        if tempC[i]>-1:
            continue
        if iwc_wsp[i]<-990:
            continue
        if iwc_ncar[i]<0.0001:
            continue
        nd_in=Nbins*ds*1e-6
        iwc=np.sum(nd_in*massB[4])
        dm=np.sum(nd_in*dbins_eq*1e3*massB[4])/iwc
        Nw=4**4/pi*iwc/(0.1*dm)**4/(rhow*1e3)
        lwc,zKu,attKu,rrate,\
            kextKu,kscatKu,g_Ku,dm_out,r_eff = psdintegrate_mass(rhow,wl[0],nd_in,\
                                                                vfallKu,dbins*1e-3,massB[4],bscatKu,\
                                                                qextKu,qscatKu,gKu)
        lwc,zKa,attKa,rrate,\
            kextKa,kscatKa,g_Ka,dm_out,r_eff = psdintegrate_mass(rhow,wl[1],nd_in,\
                                                                vfallKa,dbins*1e-3,massB[4],bscatKa,\
                                                                qextKa,qscatKa,gKa)

        lwc,zW,attW,rrate,\
            kextW,kscatW,g_W,dm_out,r_eff = psdintegrate_mass(rhow,wl[2],nd_in,\
                                                             vfallW,dbins*1e-3,massB[4],bscatW,\
                                                             qextW,qscatW,gW)
        dat1=[Nw,dm,iwc,iwc_ncar[i],tempC[i],zKu,attKu,kextKu,kscatKu,g_Ku,\
              zKa,attKa,kextKa,kscatKa,g_Ka,zW,attW,kextW,kscatW,g_W]
        dat1L.append(dat1)
        #stop
        iwcAvg=[sum(Nbins*ds*massB[k])*1e-6 for k in range(4,5)]
        
        iwcL1.append(iwc_ncar[i])
        iwcL2.append(iwcAvg)
        iwcL3.append(iwc_wsp[i])
    #break 
iwcL1=np.array(iwcL1)
iwcL2=np.array(iwcL2)
iwcL3=np.array(iwcL3)
    
    
import xarray as xr
datL=xr.DataArray(dat1L)
ds=xr.Dataset({"psd_Data":datL})
ds.attrs["title"] = "Nw,dm,iwc,iwc_ncar,tempC,zKu,attKu,kextKu,kscatKu,gKu,Ka,attKa,kextKa,kscatKa,gKa,zW,attW,kextW,kscatW,gW"
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("simulatedZ_impacts2020_3.nc", encoding=encoding)


import matplotlib.pyplot as plt
import matplotlib
plt.figure(figsize=(6,6))
c=plt.hist2d(np.log10(datL[:,0])+8,datL[:,4],bins=(3+np.arange(21)*0.5,-35+np.arange(23)*1.5),norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.ylim(-2.5,-35)
plt.xlim(5,11)
plt.xlabel("log$_{10}$(N$_w$) [m$^{-4}$]")
plt.ylabel("Temperature [$^{O}$C]")
plt.title('IMPACTS 2020')
cbar=plt.colorbar()
cbar.ax.set_title("Counts")
plt.savefig("IMPACTS_Nw.png")


#import matplotlib.pyplot as plt
#import matplotlib
plt.figure(figsize=(6,6))
c=plt.hist2d(datL[:,1],np.log10(datL[:,0])+8,bins=(0.1+np.arange(30)*0.1,3+np.arange(21)*0.5),norm=matplotlib.colors.LogNorm(),cmap='jet')
#plt.ylim(-2.5,-35)
plt.ylim(5,11)
plt.ylabel("log$_{10}$(N$_w$) [m$^{-4}$]")
plt.xlabel("D$_{m}$ [mm]")
plt.title('IMPACTS 2020')
cbar=plt.colorbar()
cbar.ax.set_title("Counts")
plt.savefig("IMPACTS_Nw_dm.png")

stop

#fig=plt.figure()
#ax=plt.subplot(111)

#plt.hist2d(iwcL1,iwcL2,bins=np.arange(30)*0.1,norm=matplotlib.colors.LogNorm(),cmap='jet')

#ax.set_aspect('equal')

#iwcL1=np.array(iwcL1)
#iwcL2=np.array(iwcL2)

iwc_m=np.zeros((30),float)
diwc=np.zeros((9,30),float)
cL=np.zeros((30),float)
dm1=iwcL2.mean(axis=0)/iwcL3.mean()

plt.hist2d(np.log10(datL[:,0])+8,datL[:,4],bins=(3+np.arange(21)*0.5,-35+np.arange(23)*1.5),norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.ylim(-2.5,-35)
plt.xlim(5,11)
#for k in range(9):
#    iwcL2[:,k]/=dm1[k]
#for i in range(30):
#    a=np.nonzero((iwcL3-i*0.1)*(iwcL3-(i+1)*0.1)<0)
#    for k in range(9):
#        iwc_m[i]=iwcL3[a].mean()
#        diwc[k,i]=(iwcL3[a]-iwcL2[a,k]).mean()
        #cL[i]=len(a[0])


#ff=Dataset("hiwrap.matched.20200207_example.nc")
#zku=ff["z_ku"][:]
#zka=ff["z_ka"][:]
#plt.figure()

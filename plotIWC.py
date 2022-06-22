from netCDF4 import Dataset
import combAlg as cAlg
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


fh=Dataset("../cloudgen/build/iwc19990827.nc")

import matplotlib.pyplot as plt

iwc=fh['iwc'][:]
z=fh['z'][:]

fhs=Dataset("/home/grecu/pyCM1v2/run/squall/cm1out.nc")
zh=fhs['zh'][:]
th=fhs['th'][-1,:,0,:]
prs=fhs['prs'][-1,:,0,:]
qv =fhs['qv'][-1,:,0,:]
tk=th*(prs/1e5)**(0.287)
tk1=tk[:,100]
prs1=prs[:,100]
rho=prs1/287/tk1

dz=z[1]-z[0]
nz=int(z[-1]/dz)
import numpy as np
lyrhgt=z[-1]-(nz-1)*dz-dz/2+np.arange(nz+1)*dz
z1dm=z[-1]-(nz-1)*dz+np.arange(nz)*dz
lyrtemp=np.interp(lyrhgt/1e3,zh,tk1)
qv1d=np.interp(z1dm/1e3,zh,qv.mean(axis=-1))
rho1d=np.interp(z1dm/1e3,zh,rho)
pres1d=np.interp(z1dm/1e3,zh,prs1)
t1d=np.interp(z1dm/1e3,zh,tk1)
ireturn=0
freq=325+5
kexttot_atm=[]

for i in range(nz):
    absair,abswv = cAlg.gasabsr98(freq,t1d[i],rho1d[i]*qv1d[i],pres1d[i],ireturn)
    kexttot_atm.append(absair+abswv)

kexttot_atm=np.array(kexttot_atm)
umu=np.cos(53/180*np.pi)
btemp=293
fisot=2.7
emis=0.9
ebar=0.9
lambert=0
salb=np.zeros((nz),float)
asym=np.zeros((nz),float)
tb = cAlg.radtran(umu,btemp,lyrtemp,lyrhgt/1e3,kexttot_atm,salb,asym,fisot,emis,ebar,lambert)
stop

import numpy as np
dmCoeffs=np.polyfit(zKuG,np.log(dmG),1)
a=np.nonzero(iwc>0.001)

nz,ny,nx=iwc.shape
zKu=np.zeros((nz,ny,nx),float)-99
dm_ice=np.zeros((nz,ny,nx),float)
dnMean=np.zeros((nz),float)
cMean=np.zeros((nz),float)

zCoeffs=np.polyfit(np.log10(gwc),zKuG,1)

import combAlg as cAlg
zKuL1,zKuL2=[],[]
for k,j,i in zip(a[0],a[1],a[2]):
    ibin=cAlg.bisection2(gwc,iwc[k,j,i])
    zKu[k,j,i]=zKuG[ibin]
    dn=(30-zKu[k,j,i])/80-0.25
    if dn<0:
        dn=0
    dn*=0.1
    dnMean[k]+=dn
    cMean[k]+=1
    ibin=cAlg.bisection2(gwc,iwc[k,j,i]/10**dn)
    zKu2=zKuG[ibin]+10*dn
    zKu[k,j,i]=zCoeffs[0]*(np.log10(iwc[k,j,i])-dn)+zCoeffs[1]+10*dn
    dm_ice[k,j,i]=np.exp(dmCoeffs[0]*(zKu[k,j,i]-10*dn)+dmCoeffs[1])
    if ibin>1:
        zKuL1.append(zKu[k,j,i])
        zKuL2.append(zKu2)

zKum=np.ma.array(zKu,mask=zKu<-98)


x=fh['x'][:]


import lidar
# INTEGER npoints,nlev,npart,ice_type
# real :: undef
#REAL pres(npoints,nlev)
#REAL presf(npoints,nlev)

dz=(z[1]-z[0])/1e3
temp=[]
pres=[]
presf=[]
q_lsliq=[]
q_lsice=[]
q_cvliq=[]
q_cvice=[]
ls_radliq=[]
ls_radice=[]
cv_radice=[]
cv_radliq=[]
for i in range(nx):
    temp1=np.interp(z/1e3,zh,tk1)
    pres1=np.interp(z/1e3,zh,prs1)
    presf1=np.interp(z[0]/1e3-dz/2+np.arange(nz+1)*dz,zh,prs1)
    rho1=np.interp(z/1e3,zh,rho)
    q_lsice1=iwc[:,128,i].data/rho1*1e-3
    q_lsice.append(q_lsice1)
    q_lsliq.append(np.zeros((nz),float))
    ls_radice.append(dm_ice[:,128,i]*1e-3)
    ls_radliq.append(np.zeros((nz),float))
    q_cvice.append(np.zeros((nz),float))
    cv_radice.append(np.zeros((nz),float))
    q_cvliq.append(np.zeros((nz),float))
    cv_radliq.append(np.zeros((nz),float))
    temp.append(temp1)
    pres.append(pres1)
    presf.append(presf1)

temp=np.array(temp)
pres=np.array(pres)
presf=np.array(presf)
undef=0.0
ice_type=1
q_lsliq=np.array(q_lsliq)
q_lsice=np.array(q_lsice)
q_cvliq=np.array(q_cvliq)
q_cvice=np.array(q_cvice)
ls_radliq=np.array(ls_radliq)
ls_radice=np.array(ls_radice)
cv_radliq=np.array(cv_radliq)
cv_radice=np.array(cv_radice)

import xarray as xr

q_lsliqX=xr.DataArray(q_lsliq)
q_lsiceX=xr.DataArray(q_lsice)
q_cvliqX=xr.DataArray(q_cvliq)
q_cviceX=xr.DataArray(q_cvice)
ls_radliqX=xr.DataArray(ls_radliq)
ls_radiceX=xr.DataArray(ls_radice)
cv_radliqX=xr.DataArray(cv_radliq)
cv_radiceX=xr.DataArray(cv_radice)
tempX=xr.DataArray(temp)
presX=xr.DataArray(pres)
presfX=xr.DataArray(presf,dims=['dim_0','dim_1_'])
d=xr.Dataset({"q_lsliq":q_lsliqX,"q_lsice":q_lsiceX,\
              "q_cvliq":q_cvliqX,"q_cvice":q_cviceX,\
              "ls_radliq":ls_radliqX,"ls_radice":ls_radiceX,\
              "cv_radliq":cv_radliqX,"cv_radice":cv_radiceX,\
              "temp":tempX,"pres":presX,"presfX":presfX})
d.to_netcdf("lidarInput_2.nc")

npart=4
nrefl=4


plt.figure()
c=plt.pcolormesh(x/1e3,z/1e3,tautot,cmap='jet')

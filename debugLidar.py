#import xarray as xr
import numpy as np
from netCDF4 import Dataset
import lidar

fh=Dataset("lidarInput.nc")

q_lsliq=fh["q_lsliq"][:]
q_lsice=fh["q_lsice"][:]
q_cvliq=fh["q_cvliq"][:]
q_cvice=fh["q_cvice"][:]
ls_radliq=fh["ls_radliq"][:]
ls_radice=fh["ls_radice"][:]
cv_radliq=fh["cv_radliq"][:]
cv_radice=fh["cv_radice"][:]
temp=fh["temp"][:]
pres=fh["pres"][:]
presf=fh["presfX"][:]

npart=4
nrefl=2
ice_type=1
undef=0.0
pmol,pnorm,pnorm_perp_tot,\
    tautot,betatot_liq,\
    betatot_ice,\
    betatot,refl, zheight = lidar.lidar_simulator(npart,nrefl,undef,\
                                         pres,presf,temp,
                                         q_lsliq,q_lsice,q_cvliq,\
                                         q_cvice,ls_radliq,\
                                         ls_radice,cv_radliq,cv_radice,\
                                         ice_type)
ix=73
bscatt1d=betatot_ice[ix,:]
salb1d=0.9+bscatt1d*0
g1d=0.7+bscatt1d*0
alt=700000
alt0=4000
nz=bscatt1d.shape[0]
extinctL=[]
for i in range(nz-1,-1,-1):
    dz=zheight[ix,i+1]-zheight[ix,i]
    if i==nz-1:
        extinct=tautot[ix,i]/dz
    else:
        extinct=(tautot[ix,i]-tautot[ix,i+1])/dz
    extinctL.append(extinct)

extinct1d=np.array(extinctL)
bscatt1d=bscatt1d[::-1]
freq=532
dr=dz
noms=0
#stop
bscatt_ms=lidar.multiscatter_lidarf(extinct1d,salb1d,g1d,bscatt1d,dr,noms,alt,alt0,freq)
import matplotlib
import matplotlib.pyplot as plt

plt.plot(pnorm[ix,::-1])
plt.plot(bscatt_ms)

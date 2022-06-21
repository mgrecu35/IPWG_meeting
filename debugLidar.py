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
    tautot,refl = lidar.lidar_simulator(npart,nrefl,undef,\
                                  pres,presf,temp,
                                  q_lsliq,q_lsice,q_cvliq,\
                                  q_cvice,ls_radliq,\
                                  ls_radice,cv_radliq,cv_radice,\
                                  ice_type)

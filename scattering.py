from numba import jit
import numpy as np
from netCDF4 import Dataset
from scipy.special import gamma as gam



def readScatProf(fname):
    fh=Dataset(fname,'r')
    temp=fh['temperature'][:]
    mass=fh['mass'][:]
    fraction=fh['fraction'][:]
    bscat=fh['bscat'][:]*4*np.pi
    Deq=1e3*(mass*1e-3*6/np.pi)**(0.333) # in mm
    ext=fh['ext'][:]
    scat=fh['scat'][:]
    g=fh['g'][:]
    vfall=fh['fall_speed'][:]
    return temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,fh

def readScatProfR(fname):
    fh=Dataset(fname,'r')
    temp=fh['temperature'][:]
    mass=fh['mass'][:]
    bscat=fh['bscat'][:]*4*np.pi
    Deq=1e3*(mass*1e-3*6/np.pi)**(0.333) # in mm
    ext=fh['ext'][:]
    vfall=fh['fall_speed'][:]
    scat=fh['scat'][:]
    g=fh['g'][:]
    #print(fh)
    #stop
    return temp,mass,bscat,Deq,ext,scat,g,vfall,fh

scatTables={}
for freq in ['13.8','35','183.31','325.15']:
    fnameIce='iceSSRG/ice-self-similar-aggregates_%s-GHz_scat.nc'%freq
    temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,fh=readScatProf(fnameIce)
    scatTables[str(freq)]=[temp,mass,fraction,bscat,Deq,ext,scat,g,vfall]
fh.close()
def interp(scatTables,freq,Dint,itemp,ifract):
    [temp,mass,fraction,bscat,Deq,ext,scat,g,vfall]=scatTables[freq]
    bscatInt=np.exp(np.interp(Dint,Deq[ifract,:]\
                              ,np.log(bscat[itemp,ifract,:])))
    extInt=np.exp(np.interp(Dint,Deq[ifract,:],\
                            np.log(ext[itemp,ifract,:])))  #m^2
    scatInt=np.exp(np.interp(Dint,Deq[ifract,:],\
                             np.log(scat[itemp,ifract,:])))  #m^2
    gInt=np.interp(Dint,Deq[ifract,:],(g[itemp,ifract,:]))  #m
    return bscatInt,scatInt,extInt,gInt

    
    
dD=0.1
Dint=np.arange(100)*dD+dD/2

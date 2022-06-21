import numpy as np


from netCDF4 import Dataset

def dm_lwc(nw,lwc,rho):
    dm=(lwc*1e-3*4**4/(nw*np.pi*rho))**(0.25)
    return dm

from scipy.special import gamma as gam

def fmu(mu):
    return 6/4**4*(4+mu)**(mu+4)/gam(mu+4)


nw=0.08
lwc=1+0.05
mu=2.0
f_mu=fmu(mu)
dm=10*dm_lwc(nw,lwc,1000)

from bhmief import bhmie
import bhmief as bh
import pytmatrix.refractive
import pytmatrix.refractive as refr

wl=[pytmatrix.refractive.wl_Ku,pytmatrix.refractive.wl_Ka,\
    pytmatrix.refractive.wl_W]

ifreq=0
refr_ind_w=pytmatrix.refractive.m_w_0C[wl[ifreq]]


rhow=1.0e3
bh.init_scatt()

ifreq=0
rhow=1000.0
rhos=400
refr_ind_s=refr.mi(wl[ifreq],rhos/rhow)


lwc_o,z,att,rrate,kext,kscat,g,\
    Nd,vfall = bh.dsdintegral(nw,f_mu,dm,mu,wl[ifreq],\
                              refr_ind_s,rhow)

stop
dD=0.1
D=np.arange(100)*dD+dD/2
qback_in=np.zeros((100),float)
qext_in=np.zeros((100),float)
qsca_in=np.zeros((100),float)
gsca_in=np.zeros((100),float)


for i,d in enumerate(D):
    sback,sext,sca,gsca=bh.getsigma_mie_w(refr_ind_s,wl[ifreq],d)
    qback_in[i]=sback
    qext_in[i]=sext
    qsca_in[i]=sca
    gsca_in[i]=gsca
    
lwc_out,z_out,att_out,\
    rrate_out,kext_out,\
    kscat_out,g_out,dm_out = bh.dsdintegrate(rhow,wl[ifreq],Nd,vfall,\
                                      D,dD,\
                                      qback_in,qext_in,qsca_in,gsca_in)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size']=12
from numba import jit



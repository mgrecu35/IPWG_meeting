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
rhos=300
refr_ind_s_Ku=refr.mi(wl[ifreq],rhos/rhow)
refr_ind_s_Ka=refr.mi(wl[ifreq+1],rhos/rhow)
mu=2.0
f_mu=fmu(mu)
lwcs=[0.001,0.002,0.003,0.004,0.005,0.01,0.02,0.03,0.04]
lwcs.extend(np.arange(400)*0.1+0.05)
gRateL,zgKuL,zgKaL=[],[],[]
attKuL,attKaL=[],[]
dmgL=[]
for lwc in lwcs:
    dm=10*dm_lwc(nw,lwc,1000)
    lwc_o,zsKu,attKu,grate,kextKu,kscatKu,gKu,\
        Nd,vfall = bh.dsdintegral_graup(nw,f_mu,dm,mu,wl[ifreq],\
                                    refr_ind_s_Ku,rhow,rhos)
    lwc_o,zsKa,attKa,grate,kextKa,kscatKa,gKa,\
        Nd,vfall = bh.dsdintegral_graup(nw,f_mu,dm,mu,wl[ifreq+1],\
                                    refr_ind_s_Ka,rhow,rhos)
    gRateL.append(grate)
    zgKuL.append(zsKu)
    zgKaL.append(zsKa)
    attKuL.append(attKu)
    attKaL.append(attKa)
    dmgL.append(dm)

import matplotlib.pyplot as plt
from netCDF4 import Dataset
try:
    fh.close()
except:
    pass


fh=Dataset("scatteringTablesGPM_SPH.nc","r+")
ng=272
zKuG=fh["zKuG"][0:272]
attKaT=np.interp(zKuG,zgKuL,attKaL)
attKuT=np.interp(zKuG,zgKuL,attKuL)
zKaT=np.interp(zKuG,zgKuL,zgKaL)
gRateLT=np.exp(np.interp(zKuG,zgKuL,np.log(gRateL)))
gwcT=np.interp(zKuG,zgKuL,lwcs)
dmgT=np.interp(zKuG,zgKuL,dmgL)


zKaG=fh["zKaG"][0:272]
attKaG=fh["attKaG"][0:272]
attKuG=fh["attKuG"][0:272]
dmG=fh["dmg"][:272]
gwc=fh["gwc"][:272]
graupRate=fh["graupRate"][:272]

fh["attKaG"][:272]=attKaT
fh["attKuG"][:272]=attKuT
fh["dmg"][:272]=dmgT
fh["gwc"][:272]=gwcT
fh["zKaG"][:272]=zKaT
fh["graupRate"][:272]=gRateLT

fh.close()
stop
zKuR=fh["zKuR"][0:272]
zKaR=fh["zKaR"][0:272]
attKaR=fh["attKaR"][0:272]
attKuR=fh["attKuR"][0:272]
dmR=fh["dmr"][:272]
rwc=fh["rwc"][:272]
rainRate=fh["rainRate"][:272]
stop
lwc_o,zw,att,rrate,kext,kscat,g,\
    Nd,vfall = bh.dsdintegral(nw,f_mu,dm,mu,wl[ifreq],\
                              refr_ind_w,rhow)


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



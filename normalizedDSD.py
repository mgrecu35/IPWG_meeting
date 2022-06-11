import numpy as np

import wrf as wrf
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from read_wrf import *
fname='/media/grecu/ExtraDrive1/wrfout_d03_2018-07-29_22:40:00'
ncFile=Dataset(fname)
it=-1
dbz=wrf.getvar(ncFile,'dbz',it)
h=wrf.getvar(ncFile,'z')
ncFile.close()
qr,qs,qg,ncr,ncs,ncg,rho,T,hf,f=read_wrf(fname,it)

rwc=qr*rho*1e3
swc=qs*rho*1e3
gwc=qg*rho*1e3
ncr=ncr*rho
ncs=ncs*rho
ncg=ncg*rho
ifreq=0

def dm_lwc(nw,lwc,rho):
    dm=(lwc*1e-3*4**4/(nw*np.pi*rho))**(0.25)
    return dm

from scipy.special import gamma as gam

def fmu(mu):
    return 6/4**4*(4+mu)**(mu+4)/gam(mu+4)


nw=0.08
lwc=np.arange(40)*0.1+0.05
mu=2.0
f_mu=fmu(mu)
dm=10*dm_lwc(nw,lwc,1000)

from bhmief import bhmie
import bhmief as bh
import pytmatrix.refractive
import pytmatrix.refractive as refr

wl=[pytmatrix.refractive.wl_Ku,pytmatrix.refractive.wl_Ka,\
    pytmatrix.refractive.wl_W]

refr_ind_w=pytmatrix.refractive.m_w_0C[wl[ifreq]]


zL=[]
attL=[]
rrateL=[]
snowRateL=[]
refr_ind_w=pytmatrix.refractive.m_w_0C[wl[ifreq]]
rhow=1000.0
rhos=500
refr_ind_s=refr.mi(wl[0],rhos/rhow)

dm2=10*dm_lwc(nw*2,lwc,1000)
for i in range(10,11):
    lwc,z,att_bh,rrate_bh,\
        kext_bh,kscat_bh,g_h =bh.dsdintegral(nw,f_mu,dm[i],mu,wl[0],\
                                             refr_ind_w,rhow)

    lwcs,zs_bh,atts_bh,snowrate_bh,\
        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(2*nw,\
                                                       f_mu,dm2[i],mu,wl[0],\
                                                       refr_ind_s,rhow,rhos)
    

i0=105
j0=221

a=np.nonzero(gwc>0.1)
dm_g=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*gwc[a]/ncg[a])**(1/3.0)

def nw_lambd(swc,nc,mu):
    rhow=1e6
    lambd=(nc*rhow*np.pi*gam(4+mu)/gam(1+mu)/6.0/swc)**(0.333)  # m-1
    n0=nc*lambd/gam(1+mu) # m-4
    n0*=1e-3 # mm-1 m-3
    #lambd*=1e-2 # cm-1
    return n0,lambd

n0,lambd=nw_lambd(gwc[a],ncg[a],mu)
dm_g2=(4+mu)/lambd
nw_g=4**4/(np.pi*rhow*1e3)*gwc[a].data/(100.0*dm_g.data)**4

#for gwc1,nw1,dm1 in zip(gwc[a],nw_g,dm_g):
#     lwcs,zs_bh,atts_bh,snowrate_bh,\
#        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_snow(nw1,\
#                                                       f_mu,dm1*1e3,mu,wl[ifreq],\
#                                                       refr_ind_s,rhow,rhos)
#     print(gwc1,lwcs,zs_bh)

piatot=0
hf=hf/9.81e3


f=2.0
rhos=900.0
refr_ind_s=refr.mi(wl[ifreq],rhos/rhow)
a1=np.nonzero(dbz[0,:,:].data>42)
piaL=[]
bh.init_scatt()

from sklearn import preprocessing
scaler  = preprocessing.StandardScaler()
from sklearn.decomposition import pca
hpca=pca.PCA()
xL=[]
for i0,j0 in zip(a1[0],a1[1]):
    x=[]
    x.extend(gwc[:,i0,j0])
    x.extend(ncg[:,i0,j0])
    x.extend(rwc[:,i0,j0])
    x.extend(ncr[:,i0,j0])
    xL.append(x)

xn=scaler.fit_transform(np.array(xL))
hpca.fit(xn)
#plt.plot(hpca.explained_variance_ratio_[0:10]*100)
from scipy.ndimage import gaussian_filter1d
zdb=[]
sfcRainRateL=[]
hsL=[]
itmax=1
tL=[]
pRateL=[]
nw_ZmL=[]
nw_g=0.08
nw_r=0.02
lwcgL=[]
zgL=[]
zrL=[]
dmL=[]
attgL=[]
attrL=[]
graupRateL=[]
rainRateL=[]
for dm_g in np.arange(10,30,0.1)*0.1:
    lwcg,zg_bh,attg_bh,grauprate_bh,\
        kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_g,\
                                                        f_mu,dm_g,mu,wl[ifreq],\
                                                        refr_ind_s,rhow,rhos)
    lwcgL.append(lwcg)
    zgL.append(zg_bh)
    attgL.append(attg_bh)
    dmL.append(dm_g)
    graupRateL.append(grauprate_bh)
    lwc,z,att_bh,rrate_bh,\
        kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dm_g,mu,wl[ifreq],\
                                             refr_ind_w,rhow)
    print(lwcg,lwc,zg_bh,z,grauprate_bh,rrate_bh)
    attrL.append(att_bh)
    rainRateL.append(rrate_bh)
    zrL.append(z)

zm=10*np.log10(0.5*10**(np.array(zrL)/10)+0.5*10**(np.array(zgL)/10))

dmmCoeff=np.polyfit(np.array(zrL)/1.,np.log10(dmL),1)
attm=0.5*(np.array(attrL)+np.array(attgL))

attRCoeffs=np.polyfit(np.array(zrL)/10.,np.log10(attrL),1)
attGCoeffs=np.polyfit(np.array(zgL)/10.,np.log10(attgL),1)
attMCoeffs=np.polyfit(np.array(zm)/10.,np.log10(attm),1)

rrateCoeffs=np.polyfit(np.array(zrL)/10.,np.log10(rainRateL),1)
grateCoeffs=np.polyfit(np.array(zgL)/10.,np.log10(graupRateL),1)
plt.plot(np.log(dmL),np.log(graupRateL))
plt.plot(np.log(dmL),np.log(rainRateL))
stop
for it in range(itmax):
    print(it)
    for i0,j0 in zip(a1[0],a1[1]):
        piatot=0
        zmL=[]
        f=1.+np.random.rand()
        f=np.exp(0.05*gaussian_filter1d(np.random.randn(61),sigma=1.5))
        fn=np.exp(0.05*gaussian_filter1d(np.random.randn(61),sigma=1.5))
        
        #print(i0,j0)
        t1=[]
        prate1=[]
        for k in range(59,-1,-1):
            t1.append(T[k,i0,j0]-273.15)
            if gwc[k,i0,j0]>0.01:
                dm_g=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*f[k]*gwc[k,i0,j0]/fn[k]/ncg[k,i0,j0])**(1/3.0)
                nw_g=4**4/(np.pi*rhow*1e3)*f[k]*gwc[k,i0,j0]/(100.0*dm_g)**4
                lwcg,zg_bh,attg_bh,grauprate_bh,\
                    kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_g,\
                                                                    f_mu,dm_g*1e3,mu,wl[ifreq],\
                                                               refr_ind_s,rhow,rhos)
                piatot+=attg_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
                if T[k,i0,j0]>265 and T[k,i0,j0]<273:
                    nw_ZmL.append([np.log10(lwcg),zg_bh])
            else:
                zg_bh=0
                grauprate_bh=0
            if swc[k,i0,j0]>0.01:
                dm_s=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*f[k]*swc[k,i0,j0]/fn[k]/ncg[k,i0,j0])**(1/3.0)
                nw_s=4**4/(np.pi*rhow*1e3)*f[k]*swc[k,i0,j0]/(100.0*dm_s)**4
                lwcs,zs_bh,atts_bh,snowrate_bh,\
                    kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_s,\
                                                                    f_mu,dm_s*1e3,mu,wl[ifreq],\
                                                               refr_ind_s,rhow,rhos)
                piatot+=atts_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
            else:
                zs_bh=0
                snowrate_bh=0
            if rwc[k,i0,j0]>0.01:
                nw_r=0.01
                dmr=10*dm_lwc(nw_r,rwc[k,i0,j0],1000)
                dm_r=(4**4/(np.pi*1e3*rhow)*f_mu*gam(mu+1)*(4+mu)**(-mu-1)*rwc[k,i0,j0]*f[k]/fn[k]/ncr[k,i0,j0])**(1/3.0)
                #print(dm_r)
                if dm_r>1.5e-3:
                    dm_r=1.5e-3
                    
                nw_r=4**4/(np.pi*rhow*1e3)*f[k]*rwc[k,i0,j0]/(100.0*dm_r)**4
                lwc,z,att_bh,rrate_bh,\
                    kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dm_r*1e3,mu,wl[ifreq],\
                                                         refr_ind_w,rhow)
                piatot+=att_bh*(hf[k+1,i0,j0]-hf[k,i0,j0])*2
            else:
                z=0
                rrate_bh=0
            zm=np.log10(10**(0.1*zs_bh)+10**(0.1*zg_bh)+10**(0.1*z))*10-0*piatot
            s='%6.2f %6.3f %6.2f %6.2f %6.3f %6.2f '%(gwc[k,i0,j0],rwc[k,i0,j0],zs_bh,z,piatot,zm)
            zmL.append(zm)
            prate1.append(rrate_bh+snowrate_bh+grauprate_bh)
        #pRateL.append(prate1)
        #print(s)
        piaL.append(piatot)
        zmint=np.interp(1.75+np.arange(60)*0.25,h[:,i0,j0]/1e3,zmL[::-1])
        tint=np.interp(1.75+np.arange(60)*0.25,h[:,i0,j0]/1e3,t1[::-1])
        prate_int=np.interp(1.75+np.arange(60)*0.25,h[:,i0,j0]/1e3,prate1[::-1])
        pRateL.append(prate_int)
        tL.append(tint)
        #plt.scatter(zmint,1.75+np.arange(60)*0.25)
        zdb.append(zmint)

        sfcRainRateL.append(rrate_bh+snowrate_bh+grauprate_bh)
        hsL.append(hf[0,i0,j0])

stop
from processWRF import processWRF


it=-2
zdb2=[]
sfcRainRate2L=[]
hs2L=[]
pia2L=[]
t2L=[]
pRate2L=[]

fname='/media/grecu/ExtraDrive1/IPHEX/wrfout_d03_2014-06-12_15:00:00'
it=4
fname='wrfout_d03_2018-07-29_22:40:00'
it=-1
processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb2,sfcRainRate2L,\
           hs2L,pia2L,t2L,pRate2L,ifreq)
stop
processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb,sfcRainRateL,\
           hsL,piaL,tL,pRateL,ifreq)

fname='wrfout_d03_2018-07-29_23:40:00'
it=0
processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb,sfcRainRateL,\
           hsL,piaL,tL,pRateL,ifreq)
it=1
processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb,sfcRainRateL,\
           hsL,piaL,tL,pRateL,ifreq)
fname='/media/grecu/ExtraDrive1/IPHEX/wrfout_d03_2014-06-12_15:00:00'
#fname='/media/grecu/ExtraDrive1/IPHEX/wrfout_d03_2014-05-23_18:54:00'
it=6
processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb,sfcRainRateL,\
           hsL,piaL,tL,pRateL,ifreq)

fname='wrfout_d03_2018-06-25_03:00:00'
it=3
processWRF(fname,it,1,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb,sfcRainRateL,\
           hsL,piaL,tL,pRateL,ifreq)




fname='wrfout_d03_2018-06-25_03:00:00'
zdb3=[]
sfcRainRate3L=[]
hs3L=[]
pia3L=[]
t3L=[]
pRate3L=[]
it=1
processWRF(fname,it,1,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb3,sfcRainRate3L,\
           hs3L,pia3L,t3L,pRate3L,ifreq)

zdb4=[]
sfcRainRate4L=[]
hs4L=[]
pia4L=[]
t4L=[]
pRate4L=[]
fname='wrfout_d03_2018-07-29_23:10:00'
it=2
processWRF(fname,it,itmax,bh,rhow,rhos,f_mu,gam,dm_lwc,mu,wl,refr_ind_s,refr_ind_w,zdb4,sfcRainRate4L,\
           hs4L,pia4L,t4L,pRate4L,ifreq)

zdb=np.array(zdb)
zdb[zdb<10]=0
zdb2=np.array(zdb2)
zdb2[zdb2<10]=0
zdb3=np.array(zdb3)
zdb3[zdb3<10]=0
zdb4=np.array(zdb4)
zdb4[zdb4<10]=0

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(30, weights='distance')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(zdb, sfcRainRateL, test_size=0.33, random_state=42)
knn.fit(X_train,y_train)
y_=knn.predict(X_test)
y2_=knn.predict(zdb2)


import xarray as xr
fname="zKuPrecip_Dataset.nc"
def writeDset(zdb,tL,piaL,sfcRainRateL,pRateL,fname):
    zKuX=xr.DataArray(zdb)
    t2cX=xr.DataArray(tL)
    piaKuX=xr.DataArray(piaL)
    sfc_pRateX=xr.DataArray(sfcRainRateL)
    pRateX=xr.DataArray(pRateL)
    dS=xr.Dataset({"piaKu":piaKuX,"zKu":zKuX,"pRate":pRateX,"t2c":t2cX})
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in dS.data_vars}
    dS.to_netcdf(fname)


writeDset(zdb,tL,piaL,sfcRainRateL,pRateL,fname)
fname="zKuPrecip_Dataset_Validation_IPHEX.nc"
writeDset(zdb2,t2L,pia2L,sfcRainRate2L,pRate2L,fname)
fname="zKuPrecip_Dataset_Validation_OKMCS.nc"
writeDset(zdb3,t3L,pia3L,sfcRainRate3L,pRate3L,fname)
fname="zKuPrecip_Dataset_Validation_ColStorm.nc"
writeDset(zdb4,t4L,pia4L,sfcRainRate4L,pRate4L,fname)


distances, indices = nbrs.kneighbors(X)

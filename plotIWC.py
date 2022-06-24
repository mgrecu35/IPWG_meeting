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
freq3=325.15+9.5
freq2=325.15+3.5
freq1=325.15+1.5


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(qv.T)
qv_sc=scaler.transform(qv.T)
from sklearn.cluster import KMeans
n_clusters=10
random_state=10
algo = KMeans(n_clusters=n_clusters, random_state=random_state)
algo.fit(qv_sc)
qvEns=[]
for qv_sc1 in algo.cluster_centers_:
    qvEns.append(qv_sc1*scaler.scale_+scaler.mean_)

kexttot_atm_Ens=[[],[],[]]
for iEns,freq in enumerate([freq1,freq2,freq3]):
    for qv1d in qvEns:
        qv1di=np.interp(z1dm/1e3,zh,qv1d)
        kexttot_atm=[]
        for i in range(nz):
            absair,abswv = cAlg.gasabsr98(freq,t1d[i],rho1d[i]*qv1di[i],pres1d[i],ireturn)
            kexttot_atm.append(absair+abswv)
        kexttot_atm=np.array(kexttot_atm)
        kexttot_atm_Ens[iEns].append(kexttot_atm)
    
kexttot_atm_Ens=np.array(kexttot_atm_Ens)
#stop
umu=np.cos(5./180*np.pi)
btemp=293
fisot=2.7
emis=0.9
ebar=0.9
lambert=0

salb=np.zeros((nz),float)
asym=np.zeros((nz),float)
tb = cAlg.radtran(umu,btemp,lyrtemp,lyrhgt/1e3,kexttot_atm,salb,asym,fisot,emis,ebar,lambert)
#stop
nz_rte=nz

import numpy as np
dmCoeffs=np.polyfit(zKuG,np.log(dmG),1)
a=np.nonzero(iwc>0.001)

nz,ny,nx=iwc.shape


zCoeffs=np.polyfit(np.log10(gwc),zKuG,1)

import combAlg as cAlg
dnz=nz_rte-nz
zKu=np.zeros((nz,ny,nx),float)-99
dm_ice=np.zeros((nz,ny,nx),float)
dnMean=np.zeros((nz),float)
cMean=np.zeros((nz),float)
fhssRG=Dataset("ssRG-scatteringTables.nc")
iwcST=fhssRG["iwc"][:]
dmST=fhssRG["dm"][:]
kextST=fhssRG["kext"][:]
kscaST=fhssRG["ksca"][:]
gST=fhssRG["g"][:]
iwcL1=[]
iwcL2=[]
nfreq=2
kexttot=np.zeros((nfreq,nz_rte),float)
kexttot_hyd=np.zeros((nfreq,nz_rte),float)
kscatot=np.zeros((nfreq,nz_rte),float)
gtot=np.zeros((nfreq,nz_rte),float)
tbL=[]
tbL_MC=[]
import MCRT
tbEns=np.zeros((3,10,256),float)
ny2=int(ny/2)
kexttot_3d=np.zeros((nfreq,nz_rte,ny,nx),float)
kscatot_3d=np.zeros((nfreq,nz_rte,ny,nx),float)
gtot_3d=np.zeros((nfreq,nz_rte,ny,nx),float)
dn2d=np.random.randn(ny,nx)*2
from scipy.ndimage import gaussian_filter

dn2d=gaussian_filter(dn2d,sigma=3)*2
dn3d=np.zeros((nz,ny,nx),float)-99
import tqdm
for i in tqdm.tqdm(range(0,nx)):
    for j in range(0,ny):
        for k in range(0,nz):
            if iwc[k,j,i]>0.0001:
                ibin=cAlg.bisection2(gwc,iwc[k,j,i])
                zKu[k,j,i]=zKuG[ibin]
                dn=(30-zKu[k,j,i])/80-0.25
                if dn<0:
                    dn=0
                dn+=dn2d[j,i]
                dn3d[k,j,i]=dn
                ibin=cAlg.bisection2(gwc,iwc[k,j,i]/10**dn)
                if ibin>271:
                    ibin=271
                    dn=np.log10(iwc[k,j,i]/gwc[271])
                zKu2=zKuG[ibin]+10*dn
                zKu[k,j,i]=zCoeffs[0]*(np.log10(iwc[k,j,i])-dn)+zCoeffs[1]+10*dn
                dm_ice[k,j,i]=np.exp(dmCoeffs[0]*(zKu[k,j,i]-10*dn)+dmCoeffs[1])
                ibin2=cAlg.bisection2(dmST[-1,:],dm_ice[k,j,i])
                iwc2=iwcST[-1,ibin2]*10**dn
                kexttot_hyd[0,dnz+k]=kextST[-1,ibin2]*10**dn
                kscatot[0,dnz+k]=kscaST[-1,ibin2]*10**dn
                gtot[0,dnz+k]=gST[-1,ibin2]
            else:
                kexttot_hyd[:,dnz+k]=0
                kscatot[:,dnz+k]=0
                gtot[:,dnz+k]=0
        kexttot_3d[:,:,j,i]=kexttot_hyd
        kscatot_3d[:,:,j,i]=kscatot
        gtot_3d[:,:,j,i]=gtot
    #stop
    #for ifreq in range(3):
    #    for iEns in range(10):
    #        kexttot=kexttot_atm_Ens[ifreq,iEns,:]+kexttot_hyd
    #        salb=kscatot/kexttot
    #        asym=gtot
    #        inc_angle=0
    #        tb = cAlg.radtran(umu,btemp,lyrtemp,lyrhgt/1e3,kexttot,salb,asym,fisot,emis,ebar,lambert)
    #        tbEns[ifreq,iEns,i]=tb
    #tb_out = MCRT.mcrt(kexttot,salb,asym,lyrtemp,lyrhgt/1e3,emis,btemp,inc_angle)
    #stop
    #print(tb,tb_out)
    #tbL.append(tb)
    #tbL_MC.append(tb_out)
            #print(iwc[k,j,i],iwc2)
            #stop
#stop

    
#zKuL1,zKuL2=[],[]
#for k,j,i in zip(a[0],a[1],a[2]):
#ibin=cAlg.bisection2(gwc,iwc[k,j,i])
#    zKu[k,j,i]=zKuG[ibin]
#    dn=(30-zKu[k,j,i])/80-0.25
#    if dn<0:
#        dn=0
#    dn*=0.1
#    dnMean[k]+=dn
#    cMean[k]+=1
#    ibin=cAlg.bisection2(gwc,iwc[k,j,i]/10**dn)
#    zKu2=zKuG[ibin]+10*dn
#    zKu[k,j,i]=zCoeffs[0]*(np.log10(iwc[k,j,i])-dn)+zCoeffs[1]+10*dn
#    dm_ice[k,j,i]=np.exp(dmCoeffs[0]*(zKu[k,j,i]-10*dn)+dmCoeffs[1])
#    if ibin>1:
#        zKuL1.append(zKu[k,j,i])
#        zKuL2.append(zKu2)

zKum=np.ma.array(zKu,mask=zKu<-98)


x=fh['x'][:]


import lidar
# INTEGER npoints,nlev,npart,ice_type
# real :: undef
#REAL pres(npoints,nlev)
#REAL presf(npoints,nlev)

dz=(z[1]-z[0])/1e3
temp=np.zeros((nz,ny,nx),float)
pres=np.zeros((nz,ny,nx),float)
presf=np.zeros((nz+1,ny,nx),float)
q_lsliq=np.zeros((nz,ny,nx),float)
q_lsice=np.zeros((nz,ny,nx),float)
q_cvliq=np.zeros((nz,ny,nx),float)
q_cvice=np.zeros((nz,ny,nx),float)
ls_radliq=np.zeros((nz,ny,nx),float)
ls_radice=np.zeros((nz,ny,nx),float)
cv_radice=np.zeros((nz,ny,nx),float)
cv_radliq=np.zeros((nz,ny,nx),float)

for i in range(nx):
    for j in range(0,ny):
        temp1=np.interp(z/1e3,zh,tk1)
        pres1=np.interp(z/1e3,zh,prs1)
        presf1=np.interp(z[0]/1e3-dz/2+np.arange(nz+1)*dz,zh,prs1)
        rho1=np.interp(z/1e3,zh,rho)
        q_lsice1=iwc[:,j,i].data/rho1*1e-3
        q_lsice[:,j,i]=(q_lsice1)
        q_lsliq[:,j,i]=(np.zeros((nz),float))
        ls_radice[:,j,i]=(dm_ice[:,j,i]*1e-3)
        ls_radliq[:,j,i]=(np.zeros((nz),float))
        q_cvice[:,j,i]=(np.zeros((nz),float))
        cv_radice[:,j,i]=(np.zeros((nz),float))
        q_cvliq[:,j,i]=(np.zeros((nz),float))
        cv_radliq[:,j,i]=(np.zeros((nz),float))
        temp[:,j,i]=(temp1)
        pres[:,j,i]=(pres1)
        presf[:,j,i]=(presf1)

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
zKuX=xr.DataArray(zKu)
presfX=xr.DataArray(presf,dims=['dim_0_1','dim_1','dim_2'])
kexttotX=xr.DataArray(kexttot_3d,dims=['nfreq','dim_0_2','dim_1','dim_2'])
kscatotX=xr.DataArray(kscatot_3d,dims=['nfreq','dim_0_2','dim_1','dim_2'])
gtotX=xr.DataArray(gtot_3d,dims=['nfreq','dim_0_2','dim_1','dim_2'])
dn3dX=xr.DataArray(dn3d,dims=['dim_0','dim_1','dim_2'])
d=xr.Dataset({"q_lsliq":q_lsliqX,"q_lsice":q_lsiceX,\
              "q_cvliq":q_cvliqX,"q_cvice":q_cviceX,\
              "ls_radliq":ls_radliqX,"ls_radice":ls_radiceX,\
              "cv_radliq":cv_radliqX,"cv_radice":cv_radiceX,\
              "temp":tempX,"pres":presX,"presfX":presfX,"zKu":zKuX,\
              "kext":kexttotX,"kscat":kscatotX,"gtot":gtotX,\
              "dn3d":dn3dX})

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in d.data_vars}
#ds.to_netcdf(filename)
d.to_netcdf("lidarInput_3d.nc", encoding=encoding)

npart=4
nrefl=4

pmol,pnorm,pnorm_perp_tot,\
    tautot,betatot_liq,\
    betatot_ice,\
    betatot,refl, zheight = \
        lidar.lidar_simulator(npart,nrefl,undef,\
                              pres[:,ny2-1,:].T,presf[:,ny2-1,:].T,temp[:,ny2-1,:].T,
                              q_lsliq[:,ny2-1,:].T,q_lsice[:,ny2-1,:].T,q_cvliq[:,ny2-1,:].T,\
                              q_cvice[:,ny2-1,:].T,ls_radliq[:,ny2-1,:].T,\
                              ls_radice[:,ny2-1,:].T,cv_radliq[:,ny2-1,:].T,\
                              cv_radice[:,ny2-1,:].T,\
                              ice_type)


#plt.figure()
#c=plt.pcolormesh(x/1e3,z/1e3,tautot,cmap='jet')
import matplotlib
matplotlib.rcParams['font.size']=12
def plotCross(jsect):
    plt.figure(figsize=(8,8))
    plt.subplot(211)
    c=plt.pcolormesh(x/1e3,z/1e3,zKum[:,jsect,:],cmap='jet')
    plt.contour(x/1e3,z/1e3,zKum[:,jsect,:],levels=[10,12],colors='black')
    c.axes.xaxis.set_visible(False)
    plt.ylabel("Height [km]")
    plt.title('Ku-band')
    cbar1=plt.colorbar(c)
    cbar1.ax.set_title('dbZ')
    plt.subplot(212)
    pnorm_perp_totm=np.ma.array(pnorm,mask=pnorm<1e-6)
    c2=plt.pcolormesh(x/1e3,z/1e3,pnorm_perp_totm.T,\
                      norm=matplotlib.colors.LogNorm(),cmap='jet')
    cbar=plt.colorbar(c2)
    cbar.ax.set_title('m$^{-1}$sr$^{-1}$')
    plt.title('Lidar')
    plt.xlabel("Distance [km]")
    plt.ylabel("Height [km]")
    plt.savefig('radarAndLidar_slice.png')

from scipy.io.idl import readsav
import numpy as np
sav_fname="aircraft_radar.sav"
sav_data = readsav(sav_fname,python_dict=True)

dtypes=sav_data['ac_rad_bf'].dtype
d1=sav_data['ac_rad_bf']
d2=sav_data['ac_all']
nbinsL=d2['conc1d_all'][0].T
dbins=d2['midbins_all'][0][:42]
dint=d2['endbins_all'][0][:43]
iwc_wsp=d2['wisper_cwc_iwc_all'][0]
iwc_all=d2['iwc_all'][0]
dm_all=d2['dmass_all'][0]
zx_tm=d2['dbz_tm0_x_all'][0]
zku_tm=d2['dbz_tm1_ku_all'][0]
zka_tm=d2['dbz_tm2_ka_all'][0]
zw_tm=d2['dbz_tm3_w_all'][0]
zku_obs=d2['ku_radar_all'][0]
zka_obs=d2['ka_radar_all'][0]
zw_obs=d2['w_radar_all'][0]
zx_obs=d2['x_radar_all'][0]
mD_Coeff=np.zeros((2),float)
mD_Coeff[0]=0.0061
mD_Coeff[1]=2.05
ds=dint[1:]-dint[0:-1]
#stop
massB=[mD_Coeff[0]*(1e-4*dbins)**(-0.2+i*0.05+mD_Coeff[1]) for i in range(9)]

iwcL=[]
iwc_wspL=[]
from scattering import *
from integrate import *
itemp=-3
ifract=14
massP=massB[4]*1e-3
msphere=np.pi*(dbins*1e-6)**3/6*980
fice=massP/msphere

bscatX,qscatX,qextX,gX,vfallX=interp_mass(scatTables,'9.6',massB[4]*1e-3,itemp,ifract,fice)
bscatKu,qscatKu,qextKu,gKu,vfallKu=interp_mass(scatTables,'13.8',massB[4]*1e-3,itemp,ifract,fice)
bscatKa,qscatKa,qextKa,gKa,vfallKa=interp_mass(scatTables,'35',massB[4]*1e-3,itemp,ifract,fice)
bscatW,qscatW,qextW,gW,vfallW=interp_mass(scatTables,'94',massB[4]*1e-3,itemp,ifract,fice)

massBF=2.94e-3*(1e-4*dbins)**1.9

#bscatX,qscatX,qextX,gX,vfallX=interp_mass(scatTables,'9.6',massBF*1e-3,itemp,ifract,fice)
#bscatKu,qscatKu,qextKu,gKu,vfallKu=interp_mass(scatTables,'13.8',massBF*1e-3,itemp,ifract,fice)
#bscatKa,qscatKa,qextKa,gKa,vfallKa=interp_mass(scatTables,'35',massBF*1e-3,itemp,ifract,fice)
#bscatW,qscatW,qextW,gW,vfallW=interp_mass(scatTables,'94',massBF*1e-3,itemp,ifract,fice)

fscale=np.exp(-0.1*1e-3*dbins)
bscatKu*=fscale
bscatKa*=fscale
bscatW*=fscale
zKuL=[]
zKaL=[]
zXL=[]
zWL=[]
rhow=1e3
import pytmatrix.refractive
wl=[pytmatrix.refractive.wl_X,pytmatrix.refractive.wl_Ku,pytmatrix.refractive.wl_Ka,pytmatrix.refractive.wl_W]
iwc_mgL=[]
dm_mgL=[]
dbins_eq=(massB[4]/rhow/1e3*6/np.pi)**(1/3.)
for i,Nbins in enumerate(nbinsL):
    nd_in=Nbins*ds*1e-6
    iwc=np.sum(nd_in*massB[4])
    if iwc>0:
        lwc,zX,attX,rrate,\
            kextX,kscatX,g_X,dm_out,r_eff = psdintegrate_mass(rhow,wl[0],nd_in,\
                                                              vfallX,dbins*1e-3,massB[4],bscatX,\
                                                              qextX,qscatX,gX)
        
        lwc,zKu,attKu,rrate,\
            kextKu,kscatKu,g_Ku,dm_out,r_eff = psdintegrate_mass(rhow,wl[1],nd_in,\
                                                                 vfallKu,dbins*1e-3,massB[4],bscatKu,\
                                                                 qextKu,qscatKu,gKu)
        lwc,zKa,attKa,rrate,\
            kextKa,kscatKa,g_Ka,dm_out,r_eff = psdintegrate_mass(rhow,wl[2],nd_in,\
                                                                 vfallKa,dbins*1e-3,massB[4],bscatKa,\
                                                                 qextKa,qscatKa,gKa)
        
        lwc,zW,attW,rrate,\
            kextW,kscatW,g_W,dm_out,r_eff = psdintegrate_mass(rhow,wl[3],nd_in,\
                                                              vfallW,dbins*1e-3,massB[4],bscatW,\
                                                              qextW,qscatW,gW)
        zKuL.append(zKu)
        zKaL.append(zKa)
        zXL.append(zX)
        zWL.append(zW)
        iwc=np.sum(nd_in*massB[4])
        dm=np.sum(nd_in*dbins_eq*1e3*massB[4])/iwc
        dm_mgL.append(dm)
        iwc_mgL.append(iwc)
    else:
        zKuL.append(-999)
        zKaL.append(-999)
        zXL.append(-999)
        zWL.append(-999)
        iwc_mgL.append(-999)
        dm_mgL.append(-999)
    if iwc_wsp[i]==iwc_wsp[i]:
        iwcL.append(iwc)
        iwc_wspL.append(iwc_wsp[i])


zKuL=np.array(zKuL)
zKaL=np.array(zKaL)
zWL=np.array(zWL)
zXL=np.array(zXL)


nbinsL=d2['conc1d_all'][0].T
dbins=d2['midbins_all'][0][:42]
dint=d2['endbins_all'][0][:43]
iwc_wsp=d2['wisper_cwc_iwc_all'][0]
iwc_all=d2['iwc_all'][0]
dm_all=d2['dmass_all'][0]
zx_tm=d2['dbz_tm0_x_all'][0]
zku_tm=d2['dbz_tm1_ku_all'][0]
zka_tm=d2['dbz_tm2_ka_all'][0]
zw_tm=d2['dbz_tm3_w_all'][0]
zku_obs=d2['ku_radar_all'][0]
zka_obs=d2['ka_radar_all'][0]
zw_obs=d2['w_radar_all'][0]
zx_obs=d2['x_radar_all'][0]

tempC=d2['tc_all'][:][0]

import xarray as xr
dm_mgLX=xr.DataArray(dm_mgL,attrs=dict(description="Dm from PSD using BF",units="mm"))
dmass_all=xr.DataArray(dm_all,attrs=dict(description="dmass ",units="same as in the IDL file"))
tempC=xr.DataArray(tempC,attrs=dict(description="temperature ",units="Celsius"))
iwc_all=xr.DataArray(iwc_all,attrs=dict(description="iwc_all",units="g/m^3"))
iwc_wisperX=xr.DataArray(iwc_wsp,attrs=dict(description="iwc_wisper",units="g/m^3"))
iwc_mgLX=xr.DataArray(iwc_mgL,attrs=dict(description="iwc from PSD using BF",units="g/m^3"))
#--------------------------------------------------------------------------------#
zkuX=xr.DataArray(zKuL,attrs=dict(description="zKu_SSRG",units="dBZ"))
zxX=xr.DataArray(zXL,attrs=dict(description="zX_SSRG",units="dBZ"))
zkaX=xr.DataArray(zKaL[:],attrs=dict(description="zKa_SSRG",units="dBZ"))
zwX=xr.DataArray(zWL[:],attrs=dict(description="zW_SSRG",units="dBZ"))
#---------------------------------------------------------------------#
zku_tmX=xr.DataArray(zku_tm,attrs=dict(description="zKu tmaxtrix",units="dBZ"))
zx_tmX=xr.DataArray(zx_tm,attrs=dict(description="zX tmatrx",units="dBZ"))
zka_tmX=xr.DataArray(zka_tm[:],attrs=dict(description="zKa tmatrix",units="dBZ"))
zw_tmX=xr.DataArray(zw_tm[:],attrs=dict(description="zW tmatrix",units="dBZ"))
#---------------------------------------------------------------------#
zku_obsX=xr.DataArray(zku_obs,attrs=dict(description="zKu observations",units="dBZ"))
zx_obsX=xr.DataArray(zx_obs,attrs=dict(description="zX observations",units="dBZ"))
zka_obsX=xr.DataArray(zka_obs,attrs=dict(description="zKa observations",units="dBZ"))
zw_obsX=xr.DataArray(zw_obs,attrs=dict(description="zW observations",units="dBZ"))

ds=xr.Dataset({"dm":dmass_all,"iwc_all":iwc_all,"iwc_wisper":iwc_wisperX,\
               "zX_SSRGA":zxX,"zKu_SSRGA":zkuX,"zKa_SSRGA":zkaX,"zW_SSRGA":zwX,\
               "zX_tm":zx_tmX,"zKu_tm":zku_tmX,"zKa_tm":zka_tmX,"zW_tm":zw_tmX,\
               "zX_obs":zx_obsX,"zKu_obs":zku_obsX,"zKa_obs":zka_obsX,"zW_obs":zw_obsX,"tempC":tempC,\
               "iwc_PSD":iwc_mgLX,"dm_PSD":dm_mgLX})

#ds.attrs["title"] = "Nw,dm,iwc,iwc_ncar,tempC,zKu,attKu,kextKu,kscatKu,gKu,Ka,attKa,kextKa,kscatKa,gKa,zW,attW,kextW,kscatW,gW"
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf("collocatedZ_SSRGA_HB.nc", encoding=encoding)


a=np.nonzero(zkuX.data>-50)
b=np.nonzero(zku_obsX.data[a]>-50)
print(np.corrcoef(zkuX.data[a][b],zku_obsX.data[a][b]))
b=np.nonzero(zka_obsX.data[a]>-50)
print(np.corrcoef(zkaX.data[a][b],zka_obsX.data[a][b]))
b=np.nonzero(zx_obsX.data[a]>-50)
print(np.corrcoef(zxX.data[a][b],zx_obsX.data[a][b]))
b=np.nonzero(zw_obsX.data[a]>-50)
print(np.corrcoef(zwX.data[a][b],zw_obsX.data[a][b]))

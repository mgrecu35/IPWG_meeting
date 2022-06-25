#import xarray as xr
import numpy as np
from netCDF4 import Dataset
import lidar

fh=Dataset("lidarInput_3d.nc",'a')

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

import numpy as np
nz,ny,nx=pres.shape
pnorm3D=np.zeros((nz,ny,nx),float)
betatot3D=np.zeros((nz,ny,nx),float)
extinct3D=np.zeros((nz,ny,nx),float)
beta_mol3D=np.zeros((nz,ny,nx),float)
tau_mol3D=np.zeros((nz,ny,nx),float)
alpha_3D=np.zeros((4,nz,ny,nx),float)
for j in range(ny):
    pmol,pnorm,pnorm_perp_tot,\
        tautot,betatot_liq,\
        betatot_ice,\
        betatot,refl, \
        zheight,\
        beta_mol, tau_mol,\
        alpha= lidar.lidar_simulator(npart,nrefl,undef,\
                                        pres[:,j,:].T,presf[:,j,:].T,\
                                        temp[:,j,:].T,
                                        q_lsliq[:,j,:].T,q_lsice[:,j,:].T,\
                                        q_cvliq[:,j,:].T,\
                                        q_cvice[:,j,:].T,\
                                        ls_radliq[:,j,:].T,\
                                        ls_radice[:,j,:].T,\
                                        cv_radliq[:,j,:].T,cv_radice[:,j,:].T,\
                                        ice_type)
    pnorm3D[:,j,:]=pnorm.T
    extinctL=[]
    for i in range(nz-1,-1,-1):
        dz=zheight[:,i+1]-zheight[:,i]
        if i==nz-1:
            extinct=tautot[:,i]/dz
        else:
            extinct=(tautot[:,i]-tautot[:,i+1])/dz
        extinctL.append(extinct)
    extinct3D[:,j,:]=np.array(extinctL)
    betatot3D[:,j,:]=betatot.T
    tau_mol3D[:,j,:]=tau_mol.T
    beta_mol3D[:,j,:]=beta_mol.T
    alpha_3D[:,:,j,:]=alpha.T
#fh.createVariable("pnorm3D","f8",("dim_0","dim_1","dim_2"))
#fh.createVariable("betatot3D","f8",("dim_0","dim_1","dim_2"))
#fh.createVariable("extinct3D","f8",("dim_0","dim_1","dim_2"))
fh["pnorm3D"][:]=pnorm3D
fh["betatot3D"][:]=betatot3D
fh["extinct3D"][:]=extinct3D
zku=fh["zKu"][:]
fh.close()
#stop

import matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.subplot(141)
for i in range(10):
    plt.plot(np.log10(betatot3D[:,i*10,100]),range(64))
    
plt.subplot(142)
for i in range(10):
    plt.plot(np.log10(beta_mol3D[:,i*10,100]),range(64))

plt.subplot(143)
for i in range(10):
    plt.plot((tau_mol3D[:,i*10,100]),range(64))

plt.subplot(144)
for i in range(10):
    plt.plot(np.log10(pnorm3D[:,i*10,100]),range(64))

plt.figure()
plt.hist2d(zku[:,:,:].T.flatten(),alpha_3D[1,:,:,:].T.flatten(),\
           bins=(-30+np.arange(30)*2,10**(-4.5+np.arange(30)*0.1)),cmap='jet')
plt.yscale('log')

plt.figure()

plt.hist2d(betatot_3D[:,:,:].T.flaten(),alpha_3D[1,:,:,:].T.flatten(),\
           bins=(10**(-4.5+np.arange(30)*0.1),10**(-4.5+np.arange(30)*0.1)),cmap='jet')
plt.yscale('log')
plt.xscale('log')
#stop
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


plt.plot(pnorm[ix,::-1])
plt.plot(bscatt_ms)
fh.close()

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

lwc_o,z,att,rrate,kext,kscat,g,\
    Nd,vfall = bh.dsdintegral(nw,f_mu,dm,mu,wl[ifreq],\
                              refr_ind_w,rhow)
dD=0.1
D=np.arange(100)*dD+dD/2
qback_in=np.zeros((100),float)
qext_in=np.zeros((100),float)
qsca_in=np.zeros((100),float)
gsca_in=np.zeros((100),float)

for i,d in enumerate(D):
    sback,sext,sca,gsca=bh.getsigma_mie_w(refr_ind_w,wl[ifreq],d)
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

xL=30000
zL=4500
dx=5
dz=75
nx=int(xL/dx)
nz=int(zL/dz)
nbins=100
sigma=1500
xc=5000
umax=2
@jit(nopython=True)
def sedimentation(Nd2D,u,vfall,rhoz,nbins,dx,dz):
    nx,nz,nbins=Nd2D.shape
    for k in range(nz-2,-1,-1):
        for i in range(1,nx-1):
            for ibin in range(0,nbins):
                numerator = vfall[ibin]/rhoz[k+1]**0.5*Nd2D[i,k+1,ibin]/dz + u[k]*Nd2D[i-1,k,ibin]/dx
                denominator = vfall[ibin]/rhoz[k]**0.5/dz + u[k]/dx
                Nd2D[i,k,ibin] = numerator/denominator
    #print(nx,ny)

x=np.arange(nx)*dx
wc_init=3.0*np.exp(-1/2*(x-xc)**2/(sigma)**2)


from initCond import *

def calc_distrib(wc_init,sigma,xc,u):
    Nd2D=np.zeros((nx,nz,nbins),float)
    att2D=np.zeros((nx,nz),float)
    x=np.arange(nx)*dx
    z_init=np.zeros((nx),float)
    for i,lwc in enumerate(wc_init):
        dm=10*dm_lwc(nw,lwc,1000)
        lwc_o,z,att,rrate,kext,kscat,g,\
            Nd,vfall = bh.dsdintegral(nw,f_mu,dm,mu,wl[ifreq],\
                              refr_ind_w,rhow)
        Nd2D[i,-1,:]=Nd
        z_init[i]=z

    z=np.arange(nz)*dz


    rhoz=np.exp(-z/10.4e3)
    sedimentation(Nd2D,u,vfall,rhoz,nbins,dx,dz)
    z2d=np.zeros((nx,nz),float)
    z2d_att=np.zeros((nx,nz),float)
    att2d=np.zeros((nx,nz),float)
    dm2d=np.zeros((nx,nz),float)
    pRate2D=np.zeros((nx,nz),float)
    
    for i in range(1,nx):
        for k in range(0,nz):
            lwc_out,z_out,att_out,\
                rrate_out,kext_out,\
                kscat_out,g_out, dm_out = bh.dsdintegrate(rhow,wl[ifreq],Nd2D[i,k,:],vfall,\
                                                          D,dD,\
                                                          qback_in,qext_in,qsca_in,gsca_in)
            z2d[i,k]=z_out
            att2d[i,k]=att_out
            pRate2D[i,k]=rrate_out/rhoz[k]**0.5
            dm2d[i,k]=dm_out
        
    piaOut=np.zeros((nx),float)
    for k in range(nz-1,-1,-1):
        piaOut+=att2d[:,k]*dz/1000.
        z2d_att[:,k]=z2d[:,k]-piaOut
        piaOut+=att2d[:,k]*dz/1000.
    
    
    return z2d,piaOut,z2d_att,dm2d,pRate2D,x,z,att2d,Nd,vfall,u, z_init,Nd2D

umax=15
z=np.arange(nz)*dz
u=umax/z[-1]*(z[-1]-z)

u=np.interp(z,zdef*1e3,u_squall)
wc_init=np.interp(x,xdef,wc_squall)
u+=-u[-1]

z2d,piaOut,z2d_att,dm2d,pRate2D,x,z,att2d,Nd,vfall,u,z_init,Nd2D=calc_distrib(wc_init,sigma,xc,u)

#@jit(nopython==True)
def characteristics(x0,z0,vfall,u,dt,tmax):
    xL=[x0]
    zL=[z0]
    t=0
    while t<tmax and z0>0:
        u_lev=np.interp([z0],z,u)[0]
        x0+=u_lev*dt
        rhoz=np.exp(-z0/10.4e3)
        z0-=vfall/rhoz**0.5*dt
        xL.append(x0)
        zL.append(z0)
        t+=dt
    return xL,zL,t

xL_25,zL_25,t_25=characteristics(5000.,4250.,vfall[25],u,5.0,600.)
xL_5,zL_5,t_5=characteristics(5000.,4250.,vfall[5],u,5.0,1800.)
xL_9,zL_9,t_9=characteristics(5000.,4250.,vfall[9],u,5.0,1800.)
charact_x=[]
for i in range(0,100):
  xL_,zL_,t_=characteristics(0.,4250.,vfall[i],u,5.0,7200.)
  x_trav=np.interp(z+dz,zL_[::-1],xL_[::-1])
  charact_x.append(x_trav)

charact_x=np.array(charact_x)
z2d_c=np.zeros((nx,nz),float)
att2d_c=np.zeros((nx,nz),float)
for ilev in range(0,10):
    for i in range(nx):
        nd_init=np.zeros((100),float)
        for ibin in range(1,100):
            xinit=x[i]-charact_x[ibin,ilev]
            if xinit>=-60e3:
                ix_init=int((xinit/dx))
                nd_init[ibin]=Nd2D[ix_init%nx,-1,ibin]
            lwc_out,z_out,att_out,\
                rrate_out,kext_out,\
                kscat_out,g_out, dm_out = bh.dsdintegrate(rhow,wl[ifreq],nd_init[:],vfall,\
                                                          D,dD,\
                                                          qback_in,qext_in,qsca_in,gsca_in)
        z2d_c[i,ilev]=z_out
        att2d_c[i,ilev]=att_out


a0=np.nonzero(z2d_c[1000:,0:10]>30)
dalpha=att2d_c[1000:,0:10][a0]/(10**(0.1*z2d_c[1000:,0:10][a0]*attKuCoeffs[0])*10**(attKuCoeffs[1]))

plt.scatter(z2d_c[1000:,0:10][a0],dalpha)
plt.ylabel("k/($\\alpha $*Z$^{\\beta})$")
plt.xlabel('Z$_{Ku}$ [dBZ]')
plt.savefig('kZ_bounded_domain.png')

stop
plt.figure()
plt.plot(np.array(xL_25)/1e3,np.array(zL_25)/1e3)
plt.plot(np.array(xL_5)/1.e3,np.array(zL_5)/1e3)
plt.plot(np.array(xL_9)/1.e3,np.array(zL_9)/1e3)
plt.xlabel('x (km)')
plt.ylabel('height (km)')
plt.title('Sedimentation characteristics')
plt.legend(['Dm=2.5 mm', 'Dm=1.0 mm','Dm=0.55 mm'])
plt.savefig('sedimentationCharact_squall.png')

fig=plt.figure(figsize=(12,8))
plt.subplot(121)
im=plt.pcolormesh(x/1000.,z/1000.,z2d.T,cmap='jet',vmin=10,vmax=50)
plt.title("Effective reflectivity")
plt.xlabel('x (km)')
plt.ylabel('z (km)')
#plt.colorbar()
plt.subplot(122)
plt.pcolormesh(x/1000.,z/1000.,z2d_att.T,cmap='jet',vmin=10,vmax=50)
plt.title("Attenuated reflectivity")
plt.xlabel('x (km)')
#plt.ylabel('z (km)')
#plt.colorbar()
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.9)
cb_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
cbar = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
cbar.ax.set_xlabel("dBZ")
plt.savefig('zKu_u_squall.png')


xw=np.arange(1001)*dx
wf=np.exp(-((2500-xw)/5000)**2*4*np.log(2))
zmL=np.zeros((nx-1000,60),float)
zm=np.zeros((60),float)
@jit(nopython=True)
def convolve_zm(zmL,zm,z2d_att,wf):
    for i in range(500,nx-500):
        zm*=0
        wt=0
        for j in range(1000):
            zm+=10**(0.1*z2d_att[i+j-500,:])*wf[j]
            wt+=wf[j]
        zm=10*np.log10(zm/wt)
        zmL[i-500,:]=zm

convolve_zm(zmL,zm,z2d_att,wf)

fig=plt.figure(figsize=(12,8))
plt.subplot(211)
im=plt.pcolormesh(x/1000.,z/1000.,z2d_att.T,cmap='jet',vmin=10,vmax=50)
plt.title("Original resolution")
plt.xlabel('x (km)')
plt.ylabel('z (km)')
im.axes.xaxis.set_visible(False)
#plt.colorbar()
plt.subplot(212)
plt.pcolormesh(x[500:-500]/1000.,z/1000.,zmL.T,cmap='jet',vmin=10,vmax=50)
plt.title("DPR resolution")
plt.ylabel('z (km)')
plt.xlabel('x (km)')
plt.xlim(x[0]/1e3,x[-1]/1e3)
#plt.ylabel('z (km)')
#plt.colorbar()
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8)
cb_ax = fig.add_axes([0.875, 0.2, 0.025, 0.6])
cbar = fig.colorbar(im, cax=cb_ax, orientation='vertical')
cbar.ax.set_title("dBZ")
plt.savefig('conv_zKu_u_squall.png')

from netCDF4 import Dataset

fh=Dataset("scatteringTablesGPM.nc")
zKu=fh["zKuR"][:]
attKu=fh["attKuR"][:]
a=np.nonzero(zKu>17)
attKuCoeffs=np.polyfit(zKu[a]/10,np.log10(attKu[a]),1)

import matplotlib
fig=plt.figure(figsize=(12,8))
ax1=plt.subplot(211)
im1=plt.pcolormesh(x/1000.,z/1000.,dm2d.T,cmap='jet',vmax=3)
#plt.xlabel('x (km)')
plt.ylabel('z (km)')
plt.title('Dm')
plt.colorbar()
plt.subplot(212)
alpha=(att2d/10**(0.1*attKuCoeffs[0]*z2d_att)/10**attKuCoeffs[1])
alpham=np.ma.array(alpha,mask=alpha<0.1)
im=plt.pcolormesh(x/1000.,z/1000.,\
                  alpham.T,cmap='jet',
                  norm=matplotlib.colors.LogNorm())
plt.title("Original resolution")
plt.xlabel('x (km)')
plt.ylabel('z (km)')
plt.colorbar()
stop
#plt.figure()
#plt.plot(10*np.log10((10**(0.1*z2d_att)+1e-3).mean(axis=0)),z/1e3)
#plt.plot(10*np.log10((10**(0.1*z2d)+1e-3).mean(axis=0)),z/1e3)

fig2=plt.figure(figsize=(12,8))
ax1=plt.subplot(121)
dm2dm=np.ma.array(dm2d,mask=pRate2D<1)
im1=plt.pcolormesh(x/1000.,z/1000.,dm2dm.T,cmap='jet',vmax=3)
plt.xlabel('x (km)')
plt.ylabel('z (km)')
plt.title('Dm')
#plt.colorbar()
ax2=plt.subplot(122)
im2=plt.pcolormesh(x/1000.,z/1000.,pRate2D.T,cmap='jet',vmax=100)
plt.xlabel('x (km)')
plt.title('Precipitation rate')
fig2.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.9)
x1,y1,l1,h1=ax1.get_position().bounds
cb_ax1 = fig2.add_axes([0.1, 0.12, l1, 0.05])
cbar1 = fig2.colorbar(im1, cax=cb_ax1, orientation='horizontal')
cbar1.ax.set_xlabel('mm')
x2,y2,l2,h2=ax2.get_position().bounds
cb_ax2 = fig2.add_axes([x2, 0.12, l2, 0.05])
cbar2 = fig2.colorbar(im2, cax=cb_ax2, orientation='horizontal')
cbar2.ax.set_xlabel('mm/h')
plt.savefig('dm_PrecipRate_usquall_15.png')
#plt.ylabel('z (km)')
#plt.colorbar()


#plt.figure()
#a=np.nonzero(z2d>10)
#plt.scatter(z2d[a],att2d[a])
#plt.yscale('log')

#plt.scatter(zKu,attKu)
stop
piaOutL=[]
import tqdm
for umax in tqdm.tqdm(range(1,16)):
    z2d,piaOut,z2d_att,dm2d,pRate2D,x,z,att2d=calc_distrib(sigma,xc,umax)
    piaOutL.append(piaOut)


fig=plt.figure(figsize=(6,4))
for pia in piaOutL:
    plt.plot(x/1e3,pia)
    
plt.xlim(0,x.max()/1e3)
plt.xlabel('x(km)')
plt.ylabel('dB')
plt.title('PIA')
plt.savefig('PIA_shear.png')
    

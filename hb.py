import numpy as np

attRCoeffs=np.array([ 0.73007628, -3.43508035])
attGCoeffs=np.array([ 0.73, -4.8185116])
attMCoeffs=np.array([ 0.7711885 , -3.79301267])

def hb(zKum,alpha,beta,dr,srt_piaKu):
    q=0.2*np.log(10)
    zeta=q*beta*alpha*10**(0.1*zKum*beta)*dr
    #srt_piaKu=4.0
    zetamax=1.-10**(-srt_piaKu/10.*beta)
    if zeta.cumsum()[-1]>zetamax:
        eps=0.9999*zetamax/zeta.cumsum()[-1]
        #zeta=eps*zeta
    else:
        eps=1.0
    corrc=eps*zeta.cumsum()
    zc=zKum-10/beta*np.log10(1-corrc)
    return zc,eps,-10/beta*np.log10(1-corrc[-1])

def profiling(zKum,srtPIA,dmgCoeff,dmrCoeff,attGCoeffs,attRCoeffs,hb,nEns,refr_ind_s,refr_ind_w,rhos,rhow,\
              wl,ifreq,nz,dr,nfz,bh,nw_g,nw_r,f_mu,mu):
    dmg=10**(dmgCoeff[0]*zKum+dmgCoeff[1])
    #plt.plot(dmg,range(nz))
    ah=np.argmax(dmg)
    beta=attRCoeffs[0]
    alpha=np.interp(range(nz),[0,60,61,86],[10**attGCoeffs[1],10**attGCoeffs[1],\
                                            1.025*10**attRCoeffs[1],1.025*10**attRCoeffs[1]])
    srt_piaKu=srtPIA
    zc,eps,pia=hb(zKum,alpha,beta,dr,srt_piaKu)
    dmg=10**(dmrCoeff[0]*zc[:]+dmrCoeff[1])
    piaKu=0
    zG=[]
    gRate=[]
    for k in range(0,nfz):
        lwcg,zg_bh,attg_bh,grauprate_bh,\
            kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_g,\
                                                            f_mu,dmg[k],mu,wl[ifreq],\
                                                            refr_ind_s,rhow,rhos)
        piaKu+=attg_bh*dr
        zG.append(zg_bh-piaKu)
        piaKu+=attg_bh*dr
        gRate.append(grauprate_bh)
    dx=0
    zKuREns=[]
    pRateEns=[]
    dmrEns=[]
    zTopEns=[]
    for iEns in range(nEns):
        x=0.5*np.random.randn()
        dmr1=1.05*dmg[-1]*np.exp(x)
        lwc,z,att_bh,rrate_bh,\
            kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dmr1,mu,wl[ifreq],\
                                                 refr_ind_w,rhow)

        lwcg,zg_bh,attg_bh,grauprate_bh,\
            kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_g,\
                                                            f_mu,dmr1,mu,wl[ifreq],\
                                                            refr_ind_s,rhow,rhos)
        fR=0.25
        zp=10*np.log10(fR*10**(0.1*z)+(1-fR)*(0.1*zg_bh))
        zTopEns.append(zp)
        dmrEns.append(dmr1)
    #covXY=np.cov(dmrEns,zTopEns)
    #gain=covXY[0,1]/(covXY[1,1]+1)
    #dmgS=np.mean(dmrEns)+kgain*(zKum[nfz]-np.mean(zTopEns))
    #print(dmrEns)
    #print(zTopEns)
    #print(zKum[nfz],np.mean(zTopEns))
    #dmr1=dmgS
    #lwc,z,att_bh,rrate_bh,\
    #    kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dmr1,mu,wl[ifreq],\
    #                                         refr_ind_w,rhow)
    #print(z,kgain)
    #stop
    dmgS=dmg[-1]
    for iEns in range(nEns):
        x=0.5*np.random.randn()
        piaR=0
        zKu1=[]
        pRate1=[]
        dx=0
        for k in range(nfz,nz):
            if np.random.randn()>0:
                dx=dx+0.05
            else:
                dx=dx-0.05
            x=0.9*x+0.1*np.random.randn()
            dmr1=1.0*dmgS*np.exp(x)
            dmr1=dmg[k]
            lwc,z,att_bh,rrate_bh,\
                kext_bh,kscat_bh,g_h =bh.dsdintegral(nw_r,f_mu,dmr1,mu,wl[ifreq],\
                                                     refr_ind_w,rhow)
            lwcg,zg_bh,attg_bh,grauprate_bh,\
                kexts_bh,kscats_bh,gs_bh = bh.dsdintegral_graup(nw_g,\
                                                                f_mu,dmr1,mu,wl[ifreq],\
                                                                refr_ind_s,rhow,rhos)
            if k-nfz<3:
                fR=(k-nfz+1)*0.25
                #print(fR)
                
            else:
                fR=1
            fR=1.
            z=10*np.log10(fR*10**(0.1*z)+(1-fR)*(0.1*zg_bh))
            att_bh=fR*att_bh+(1-fR)*attg_bh
            piaR+=att_bh*dr
            zKu1.append(z-piaR)
            piaR+=att_bh*dr
            pRate1.append(fR*rrate_bh+(1-fR)*grauprate_bh)
        pRateEns.append(pRate1)
        zKuREns.append(zKu1)
    return zG,zKuREns,pRateEns

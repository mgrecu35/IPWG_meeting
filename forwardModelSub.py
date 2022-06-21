import numpy as np


def computeZKu(pRate1,dng,graupRate,attKaG,attKuG,rainRate,attKaR,attKuR,\
               zKuG,zKuR,cAlg):
    zKu1=[]
    piaKu=0
    piaKa=0
    for k,prate1 in enumerate(pRate1[:40]):
        if prate1>0:
            ibin=cAlg.bisection2(graupRate[:272],prate1/10**dng[k])
            ibin=min(271,ibin)
            piaKa+=10**dng[k]*attKaG[ibin]*0.125
            piaKu+=10**dng[k]*attKuG[ibin]*0.125
            zku1=zKuG[ibin]+10*dng[k]-piaKu
            piaKu+=10**dng[k]*attKuG[ibin]*0.125
            piaKa+=10**dng[k]*attKaG[ibin]*0.125
            zKu1.append(zku1)
            #if k==40:
                #print("graup",zku1,10**dng[k]*attKuG[ibin]*0.125,ibin)
        else:
            zKu1.append(0)
    for k,prate1 in enumerate(pRate1[40:45]):
        if prate1>0:
            ibinR=cAlg.bisection2(rainRate[:289],prate1/10**dng[k+40])
            ibinR=min(288,ibinR)
            attKaR1=10**dng[k+40]*attKaR[ibinR]*0.125
            attKuR1=10**dng[k+40]*attKuR[ibinR]*0.125
            zku1R=zKuR[ibinR]+10*dng[k+40]

            ibinG=cAlg.bisection2(graupRate[:272],prate1/10**dng[k+40])
            ibinG=min(271,ibinG)
            attKaG1=10**dng[k+40]*attKaG[ibinG]*0.125
            attKuG1=10**dng[k+40]*attKuG[ibinG]*0.125
            zku1G=zKuG[ibinG]+10*dng[k+40]
            f=(k+1)/5.
            piaKu+=(1-f)*attKuG1+f*attKuR1
            zku1=10*np.log10((1-f)*10**(0.1*zku1G)+f*10**(0.1*zku1R))-piaKu
            #print(zku1G,attKuG1,ibinG)
            piaKu+=(1-f)*attKuG1+f*attKuR1
            zKu1.append(zku1)
        else:
            zKu1.append(0.0)
    for k,prate1 in enumerate(pRate1[45:]):
        if prate1>0:
            ibin=cAlg.bisection2(rainRate[:289],prate1/10**dng[k+45])
            ibin=min(288,ibin)
            piaKa+=10**dng[k+45]*attKaR[ibin]*0.125
            piaKu+=10**dng[k+45]*attKuR[ibin]*0.125
            zku1=zKuR[ibin]+10*dng[k+45]-piaKu
            piaKu+=10**dng[k+45]*attKuR[ibin]*0.125
            piaKa+=10**dng[k+45]*attKaR[ibin]*0.125
            zKu1.append(zku1)
        else:
            zKu1.append(0.0)
    return zKu1,piaKu,piaKa

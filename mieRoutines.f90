module scatteringTables
  real :: qback(100),qext(100),qsca(100),gsca(100)
  real :: qback_g(100),qext_g(100),qsca_g(100),gsca_g(100)
  integer :: igraup, irain
end module scatteringTables

subroutine getsigma_mie_w(refr_ind,wl1,d,sback,sext,sca,gsca)
  complex:: refr_ind
  real :: wl1, d
  real, intent(out) :: sback, sext, sca, gsca
  real :: x, qext, qsca, qback
  integer:: MXNANG,NMXX, nang
  PARAMETER(MXNANG=100,NMXX=15000)
  complex:: S1(2*MXNANG-1),S2(2*MXNANG-1)
  real :: pi
  pi=atan(1.0)*4
  x=d*pi/wl1
  nang=89
  !s1,s2,qext,qsca,qback,gsca=bhmie(x,refr_ind,nang)
  call bhmie(x,refr_ind,nang,s1,s2,qext,qsca,qback,gsca)
  sback=(qback*pi/4*(d)**2)
  sext=(qext*pi/4*(d)**2)
  sca=(qsca*pi/4*(d)**2)

end subroutine getsigma_mie_w

subroutine init_scatt()
  use scatteringTables
  igraup=0
  irain=0
end subroutine init_scatt
subroutine dsdIntegral(nw,f_mu,dm,mu,wl,refr_ind,rho,&
     lwc,Z,att,rrate,kext,kscat,g,Nd_out,vfall_out)
  use scatteringTables
  implicit none
  real :: d(100), dD
  real :: lambd, vfall(100), zFact, rho, pi
  real :: nw, f_mu, dm, wl, mu
  complex :: refr_ind
  real,intent(out) :: lwc, Z, att,rrate, kext, kscat, g
  integer :: i
  real, intent(out):: Nd_out(100),vfall_out(100)
  !real :: qback, qext, qsca, gsca
  pi=atan(1.0)*4
  dD=0.1
  do i=0,99
     d(i+1)=i*dD+dD/2
     vfall(i+1)=3.78*d(i+1)**0.67
     vfall(i+1)=-0.1021+4.932*d(i+1)-0.9551*d(i+1)**2+0.07934*d(i+1)**3-&
          0.002362*d(i+1)**4
  enddo
  lwc=0
  lambd=(4+mu)/(dm)
  
  
  Z=0
  zFact=wl**4/pi**5
  att=0
  rrate=0
  kext=0
  kscat=0
  g=0
  do i=1,100
     lwc=lwc+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*(0.1*d(i))**3/6&
          *pi*dD/10*rho*1e3
     !if(irain.eq.0) then
     call getsigma_mie_w(refr_ind,wl,d(i),qback(i),qext(i),qsca(i),gsca(i))
     !endif
     Z=Z+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*zFact*qback(i)*1e6
     att=att+4.343*nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qext(i)*1e3
     kext=kext+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qext(i)*1e3
     kscat=kscat+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qsca(i)*1e3
     g=g+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qsca(i)*gsca(i)*1e3
     rrate=rrate+nw*f_mu*(d(i)/dm)**mu*&
          exp(-lambd*d(i))*(0.1*d(i))**3*dD/10*pi/6*vfall(i)*3.6e6
     Nd_out(i)=nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*1e6
     vfall_out(i)=vfall(i)
  end do
  Z=log10(Z)*10
  g=g/kscat
  !irain=1
end subroutine dsdIntegral

subroutine dsdIntegrate(rho,wl,&
     lwc,Z,att,rrate,kext,kscat,g,Nd_in,vfall_in,d,dD,&
     qback_in,qext_in,qsca_in,gsca_in,dm_out, r_eff)
  use scatteringTables
  implicit none
  real :: lambd, vfall(100), zFact, rho, pi
  real :: nw, f_mu, dm, wl, mu
  complex :: refr_ind
  real,intent(out) :: lwc, Z, att,rrate, kext, kscat, g
  integer :: i
  real, intent(in):: Nd_in(100),vfall_in(100), d(100), dD
  real,intent(in):: qback_in(100),qext_in(100),qsca_in(100),gsca_in(100)
  real, intent(out) :: dm_out, r_eff
  real :: mom2
  !real :: qback, qext, qsca, gsca
  pi=atan(1.0)*4

  lwc=0
  
  Z=0
  zFact=wl**4/pi**5
  att=0
  rrate=0
  kext=0
  kscat=0
  g=0
  dm_out=0
  !print*,zFact,'int'
  !print*, qback
  r_eff=0
  mom2=0
  do i=1,100
     lwc=lwc+Nd_in(i)/1e6*(0.1*d(i))**3/6&
          *pi*rho*1e3
     dm_out=dm_out+Nd_in(i)/1e6*(0.1*d(i))**3/6&
          *pi*rho*1e3*d(i)
     r_eff=r_eff+Nd_in(i)/1e6*(0.1*d(i))**2/6&
          *pi*rho*1e3*d(i)
     mom2=mom2+Nd_in(i)/1e6*(0.1*d(i))**2/6&
          *pi*rho*1e3
     !call getsigma_mie_w(refr_ind,wl,d(i),qback(i),qext(i),qsca(i),gsca(i))
     Z=Z+Nd_in(i)*zFact*qback_in(i)
     att=att+4.343*Nd_in(i)*qext_in(i)*1e3/1e6
     kext=kext+Nd_in(i)*qext_in(i)*1e3/1e6
     kscat=kscat+nw*Nd_in(i)*qsca_in(i)*1e3/1e6
     g=g+Nd_in(i)*qsca_in(i)*gsca_in(i)*1e3/1e6
     rrate=rrate+Nd_in(i)*(0.1*d(i))**3*pi/6*vfall_in(i)*3.6e6/1e6
  end do
  Z=log10(Z)*10
  if(lwc>0) then
     dm_out=dm_out/lwc
     r_eff=r_eff/mom2
  endif
  g=g/kscat
end subroutine dsdIntegrate


subroutine dsdIntegral_graup(nw,f_mu,dm,mu,wl,refr_ind_s,rho,rhos,&
     lwc,Z,att,rrate,kext,kscat,g,Nd_out,vfall_out)
  use scatteringTables
  implicit none
  real :: d(100), dD
  real :: lambd, vfall, zFact, rho, rhos, pi
  real :: nw, f_mu, dm, wl, mu
  complex :: refr_ind_s
  real,intent(out) :: lwc, Z, att,rrate, kext, kscat, g
  real,intent(out) :: Nd_out(100), vfall_out(100)
  integer :: i
  !real :: qback, qext, qsca, gsca
  real :: ds
  pi=atan(1.0)*4
  dD=0.1
  do i=0,99
     d(i+1)=i*dD+dD/2
  enddo
  lwc=0
  lambd=(4+mu)/(dm)
  
  
  Z=0
  zFact=wl**4/pi**5
  !print*,zFact,'graup'
  att=0
  rrate=0
  kext=0
  kscat=0
  g=0
  do i=1,100
     lwc=lwc+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*(0.1*d(i))**3/6&
          *pi*dD/10*rho*1e3
     ds=d(i)*(rho/rhos)**(1.0/3.0)
     if(igraup.eq.0) then
        call getsigma_mie_w(refr_ind_s,wl,ds,qback_g(i),qext_g(i),qsca_g(i),gsca_g(i))
     endif
     Z=Z+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*zFact*qback_g(i)*1e6
     att=att+4.343*nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qext_g(i)*1e3
     kext=kext+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qext_g(i)*1e3
     kscat=kscat+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qsca_g(i)*1e3
     g=g+nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*qsca(i)*gsca_g(i)*1e3
     vfall=4.88*(0.1*ds)**0.84
     rrate=rrate+nw*f_mu*(d(i)/dm)**mu*&
          exp(-lambd*d(i))*(0.1*d(i))**3*dD/10*pi/6*vfall*3.6e6
     Nd_out(i)=nw*f_mu*(d(i)/dm)**mu*exp(-lambd*d(i))*dD/10*1e6
     vfall_out(i)=vfall
  end do
  !print*, qback_g
  Z=log10(Z)*10
  g=g/kscat
  igraup=0
end subroutine dsdIntegral_graup


!def dsdIntegralSnow(nw,f_mu,dm,mu,wl,refr_ind,rhos,rho):
!    dD=0.1
!    d=np.arange(100)*dD+dD/2

!    lwc=0
!    lambd=(4+mu)/(dm)

!    lwc=nw*f_mu*(d/dm)**mu*np.exp(-lambd*d)*(0.1*d)**3/6*np.pi*dD/10*rho*1e3
!    Z=0
!    zFact=wl**4/np.pi**5
!    att=0
!    rrate=0
!    kext=0
!    kscat=0
!    g=0
!    refr_ind_s=refr.mi(wl,rhos/rho)
!    for i in range(100):
!        ds=d[i]*(rho/rhos)**(1.0/3.0)
!        mieProp=getsigma_mie_w(refr_ind_s,wl,ds)
!        qback=mieProp[0]
!        qext=mieProp[1]
!        qsca=mieProp[2]
!        gsca=mieProp[3]
!        Z+=nw*f_mu*(d[i]/dm)**mu*np.exp(-lambd*d[i])*dD/10*zFact*qback*1e6
!        att+=4.343*nw*f_mu*(d[i]/dm)**mu*np.exp(-lambd*d[i])*dD/10*qext*1e3
!        kext+=nw*f_mu*(d[i]/dm)**mu*np.exp(-lambd*d[i])*dD/10*qext*1e3
!        kscat+=nw*f_mu*(d[i]/dm)**mu*np.exp(-lambd*d[i])*dD/10*qsca*1e3
!        g+=nw*f_mu*(d[i]/dm)**mu*np.exp(-lambd*d[i])*dD/10*qsca*gsca*1e3
!        vfall=4.88*(0.1*ds)**0.84
!        rrate=rrate+nw*f_mu*(d[i]/dm)**mu*\
!            np.exp(-lambd*d[i])*(0.1*d[i])**3*dD/10*np.pi/6*vfall*3.6e6
!    return lwc.sum(),np.log10(Z)*10, att,rrate, kext, kscat, g/kscat

!  
  

subroutine multiscatter_lidarf(nrange,extinct,salb,g,bscatt,bscatt_ms,dr,noMS,&
     alt,alt0,freq)
implicit none
integer :: nrange,noMS,i,noNorm
real :: extinct(nrange),salb(nrange),g(nrange),ext2bscatt(nrange), lambd
real , intent(out) :: bscatt_ms(nrange)
real :: bscatt(nrange)
real :: alt,alt0,dr,theta,freq

real :: pi4, pi, lamb4
!#  float lamb=0.00857;
!#  float kextFort[88],z35med[88], ext2bscattFort[88], salbFort[88], gFort[88]; 
!#  float pi4=97.409091034;
!#  float lamb4=5.39415e-9;
!#  float Z=pow(10.,0.1*z35med[i])*pi4/1e18/lamb4/4*0.93;
integer :: k
real :: dz


lambd=0.532e-6
ext2bscatt=1e6
do k=1,nrange
   if (bscatt(k)>1e-9) then
      ext2bscatt(k)=extinct(k)/bscatt(k)
   endif
end do

print*, salb
print*, g
call multiscatter_lidar(nrange, extinct, ext2bscatt, salb, g, &
     bscatt_ms, lambd,noMS,theta,dr,alt,alt0)

!int multiscatter_lidar_(int *nrange, float *extFort, 
!  float *ext2bscatt, float *salbFort, float *gFort,
!  float *bscatFort, float *lambd, int *noMS, float *angle, float *dr,
!  float *alt)
!multiscatter_(int *nrange, float *extFort, 


end subroutine multiscatter_lidarf


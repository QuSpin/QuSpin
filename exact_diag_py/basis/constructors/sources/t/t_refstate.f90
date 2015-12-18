subroutine s_RefState_T(N,basis,col,ME,Ns,L,kblock,a)
implicit none
real(kind=4), parameter :: pi=3.14159265359
complex(kind=4), parameter :: i=cmplx(0.0,1.0)
integer(kind=4), intent(in) :: Ns, L, kblock, a
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=4), intent(inout), dimension(Ns) :: ME
real(kind=4) :: k
integer(kind=4), external :: FindZstate
integer(kind=4) :: i1,i2,r,s,j,ll


k=(2*pi*a*kblock)/L


do i1=1,Ns

ll=0
r=col(i1)
s=col(i1)
do i2=1,L
r=ishftc(r,-a,L)
if(r.lt.s) then
ll=i2;s=r;
end if
end do

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i1)=ME(i1)*sqrt(float(N(i1))/float(N(j)))*exp(-i*k*ll)
col(i1)=j
else
ME(i1)=0.0
col(i1)=i1
endif

end do

end subroutine










subroutine d_RefState_T(N,basis,col,ME,Ns,L,kblock,a)
implicit none
real(kind=8), parameter :: pi=3.14159265359d0
complex(kind=8), parameter :: i=cmplx(0.0,1.0d0)
integer(kind=4), intent(in) :: Ns, L, kblock, a
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=8), intent(inout), dimension(Ns) :: ME
real(kind=8) :: k
integer(kind=4), external :: FindZstate
integer(kind=4) :: i1,i2,r,s,j,ll


k=(2*pi*a*kblock)/L


do i1=1,Ns

ll=0
r=col(i1)
s=col(i1)
do i2=1,L
r=ishftc(r,-a,L)
if(r.lt.s) then
ll=i2;s=r;
end if
end do

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i1)=ME(i1)*dsqrt(dble(N(i1))/dble(N(j)))*zexp(-i*k*ll)
col(i1)=j
else
ME(i1)=0.0d0
col(i1)=i1
endif

end do

end subroutine










subroutine c_RefState_T(N,basis,col,ME,Ns,L,kblock,a)
implicit none
real(kind=4), parameter :: pi=3.14159265359
complex(kind=4), parameter :: i=cmplx(0.0,1.0)
integer(kind=4), intent(in) :: Ns, L, kblock, a
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=4), intent(inout), dimension(Ns) :: ME
real(kind=4) :: k
integer(kind=4), external :: FindZstate
integer(kind=4) :: i1,i2,r,s,j,ll


k=(2*pi*a*kblock)/L


do i1=1,Ns

ll=0
r=col(i1)
s=col(i1)
do i2=1,L
r=ishftc(r,-a,L)
if(r.lt.s) then
ll=i2;s=r;
end if
end do

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i1)=ME(i1)*sqrt(float(N(i1))/float(N(j)))*exp(-i*k*ll)
col(i1)=j
else
ME(i1)=0.0
col(i1)=i1
endif

end do

end subroutine










subroutine z_RefState_T(N,basis,col,ME,Ns,L,kblock,a)
implicit none
real(kind=8), parameter :: pi=3.14159265359d0
complex(kind=8), parameter :: i=cmplx(0.0,1.0d0)
integer(kind=4), intent(in) :: Ns, L, kblock, a
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=8), intent(inout), dimension(Ns) :: ME
real(kind=8) :: k
integer(kind=4), external :: FindZstate
integer(kind=4) :: i1,i2,r,s,j,ll


k=(2*pi*a*kblock)/L


do i1=1,Ns

ll=0
r=col(i1)
s=col(i1)
do i2=1,L
r=ishftc(r,-a,L)
if(r.lt.s) then
ll=i2;s=r;
end if
end do

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i1)=ME(i1)*dsqrt(dble(N(i1))/dble(N(j)))*zexp(-i*k*ll)
col(i1)=j
else
ME(i1)=0.0
col(i1)=i1
endif

end do

end subroutine












subroutine s_RefState_Z(N,basis,col,ME,Ns,L,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=4), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,r,s,j

do i=1,Ns

g=0
r=col(i)
s=col(i)
call flip_all(r,L)

if(r.lt.s) then
g=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.gt.0) then
ME(i)=ME(i)*(zblock**g)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine









subroutine d_RefState_Z(N,basis,col,ME,Ns,L,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=8), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,r,s,j

do i=1,Ns

g=0
r=col(i)
s=col(i)
call flip_all(r,L)

if(r.lt.s) then
g=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*dsqrt(dfloat(N(j))/dfloat(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine








subroutine c_RefState_Z(N,basis,col,ME,Ns,L,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=4), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,r,s,j

do i=1,Ns

g=0
r=col(i)
s=col(i)
call flip_all(r,L)

if(r.lt.s) then
g=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine













subroutine z_RefState_Z(N,basis,col,ME,Ns,L,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=8), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,r,s,j

do i=1,Ns

g=0
r=col(i)
s=col(i)
call flip_all(r,L)

if(r.lt.s) then
g=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*dsqrt(dfloat(N(j))/dfloat(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine


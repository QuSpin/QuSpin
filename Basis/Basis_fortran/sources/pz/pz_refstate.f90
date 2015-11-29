subroutine s_RefState_PZ(N,basis,col,ME,Ns,L,pzblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, pzblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=4), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,gq,r,s,j

do i=1,Ns

gq=0
r=col(i)
s=col(i)
call flip_lr(r,L)
call flip_all(r,L)

if(r.lt.s) then
gq=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(pzblock**gq)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine









subroutine d_RefState_PZ(N,basis,col,ME,Ns,L,pzblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, pzblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=8), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,gq,r,s,j

do i=1,Ns

gq=0
r=col(i)
s=col(i)
call flip_lr(r,L)
call flip_all(r,L)

if(r.lt.s) then
gq=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(pzblock**gq)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine








subroutine c_RefState_PZ(N,basis,col,ME,Ns,L,pzblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, pzblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=4), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,gq,r,s,j

do i=1,Ns

gq=0
r=col(i)
s=col(i)
call flip_lr(r,L)
call flip_all(r,L)

if(r.lt.s) then
gq=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(pzblock**gq)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine













subroutine z_RefState_PZ(N,basis,col,ME,Ns,L,pzblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, pzblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=8), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,gq,r,s,j

do i=1,Ns

gq=0;
r=col(i)
s=col(i)
call flip_lr(r,L)
call flip_all(r,L)

if(r.lt.s) then
gq=1;s=r;
end if

j=FindZstate(basis,Ns,s)

if(j.ge.0) then
ME(i)=ME(i)*(pzblock**gq)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine


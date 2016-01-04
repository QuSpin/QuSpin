subroutine s_RefState_P_Z(N,basis,col,ME,Ns,L,pblock,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L,pblock,zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=4), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,q,r,s,t,j

do i=1,Ns
g=0;q=0;
s=col(i)
r=s
t=s
call flip_all(t,L)

if(t.lt.r) then
g=1;q=0;r=t;
end if

t=s
call flip_lr(t,L)

if(t.lt.r) then
q=1;g=0;r=t;
end if

call flip_all(t,L)
if(t.lt.r) then
q=1;g=1;r=t;
end if

j=FindZstate(basis,Ns,r)

if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*(pblock**q)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine









subroutine d_RefState_P_Z(N,basis,col,ME,Ns,L,pblock,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L,pblock,zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
real(kind=8), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,q,r,s,t,j

do i=1,Ns
g=0;q=0;
s=col(i)
r=s
t=s
call flip_all(t,L)

if(t.lt.r) then
g=1;q=0;r=t;
end if

t=s
call flip_lr(t,L)

if(t.lt.r) then
q=1;g=0;r=t;
end if

call flip_all(t,L)
if(t.lt.r) then
q=1;g=1;r=t;
end if

j=FindZstate(basis,Ns,r)
if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*(pblock**q)*dsqrt(dble(N(j))/dble(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine








subroutine c_RefState_P_Z(N,basis,col,ME,Ns,L,pblock,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L,pblock,zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=4), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,q,r,s,t,j

do i=1,Ns
g=0;q=0;
s=col(i)
r=s
t=s
call flip_all(t,L)

if(t.lt.r) then
g=1;q=0;r=t;
end if

t=s
call flip_lr(t,L)

if(t.lt.r) then
q=1;g=0;r=t;
end if

call flip_all(t,L)
if(t.lt.r) then
q=1;g=1;r=t;
end if

j=FindZstate(basis,Ns,r)

if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*(pblock**q)*sqrt(float(N(j))/float(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine













subroutine z_RefState_P_Z(N,basis,col,ME,Ns,L,pblock,zblock)
implicit none
integer(kind=4), intent(in) :: Ns, L, pblock, zblock
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=1), intent(in), dimension(Ns) :: N
complex(kind=8), intent(inout), dimension(Ns) :: ME
integer(kind=4), external :: FindZstate
integer(kind=4) :: i,g,q,r,s,t,j

do i=1,Ns
g=0;q=0;
s=col(i)
r=s
t=s
call flip_all(t,L)

if(t.lt.r) then
g=1;q=0;r=t;
end if

t=s
call flip_lr(t,L)

if(t.lt.r) then
q=1;g=0;r=t;
end if

call flip_all(t,L)
if(t.lt.r) then
q=1;g=1;r=t;
end if

j=FindZstate(basis,Ns,r)

if(j.ge.0) then
ME(i)=ME(i)*(zblock**g)*(pblock**q)*dsqrt(dble(N(j))/dble(N(i)))
col(i)=j
else
ME(i)=0.0
col(i)=i
endif

end do

end subroutine


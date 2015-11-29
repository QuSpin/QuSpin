subroutine s_SpinOp(basis,Ns,opstr,indx,N_indx,ME,col,error)
implicit none
integer(kind=4), intent(in) :: Ns,N_indx
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=4), intent(in), dimension(N_indx) :: indx
character, intent(in), dimension(N_indx) :: opstr
real(kind=4), intent(out), dimension(Ns) :: ME
integer(kind=4), intent(out), dimension(Ns) :: col
integer(kind=4), intent(out) :: error
integer(kind=4), external :: FindZstate
complex(kind=4) :: M_E
integer(kind=4) :: i,j,r,a

error=0
do i=1,Ns
M_E=1.0; r=basis(i)
do j=1,N_indx
a = iand(ishft(r,-indx(j)),1)
if(opstr(j) .eq. "c") then
cycle

else if(opstr(j) .eq. "z") then
M_E = M_E * (-1)**(a+1)*0.5

else if(opstr(j) .eq. "x") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * 0.5

else if(opstr(j) .eq. "y") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * (-1)**(a)*cmplx(0.0,0.5)

else if(opstr(j) .eq. "+") then
if (a .eq. 1) then 
col(i)=basis(i); ME=0.0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else if(opstr(j) .eq. "-") then
if (a .eq. 0) then 
col(i)=basis(i); ME=0.0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else
ME=0;col=0;error=1
return 

end if

end do

if(aimag(M_E) .ne. 0.0) then
ME=0;col=0;error=-1
return 
end if

!print*, opstr,M_E,r
ME(i)=real(M_E)
col(i)=r
end do


end subroutine






subroutine d_SpinOp(basis,Ns,opstr,indx,N_indx,ME,col,error)
implicit none
integer(kind=4), intent(in) :: Ns,N_indx
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=4), intent(in), dimension(N_indx) :: indx
character, intent(in), dimension(N_indx) :: opstr
complex(kind=8), intent(out), dimension(Ns) :: ME
integer(kind=4), intent(out), dimension(Ns) :: col
integer(kind=4), intent(out) :: error
integer(kind=4), external :: FindZstate
complex(kind=8) :: M_E
integer(kind=4) :: i,j,r,a

error=0
do i=1,Ns
M_E=1.0d0; r=basis(i)
do j=1,N_indx
a = iand(ishft(r,-indx(j)),1)
if(opstr(j) .eq. "c") then
cycle

else if(opstr(j) .eq. "z") then
M_E = M_E * (-1)**(a+1)*0.5d0

else if(opstr(j) .eq. "x") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * 0.5d0

else if(opstr(j) .eq. "y") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * (-1)**(a)*dcmplx(0.0,0.5d0)

else if(opstr(j) .eq. "+") then
if (a .eq. 1) then 
col(i)=i; ME=0.0d0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else if(opstr(j) .eq. "-") then
if (a .eq. 0) then 
col(i)=i; ME=0.0d0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else
ME=0;col=0;error=1
return 

end if

end do

if(dimag(M_E) .ne. 0.0d0) then
ME=0;col=0;error=-1
return 
end if

ME(i)=dble(M_E)
col(i)=r
end do

end subroutine







subroutine c_SpinOp(basis,Ns,opstr,indx,N_indx,ME,col,error)
implicit none
integer(kind=4), intent(in) :: Ns,N_indx
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=4), intent(in), dimension(N_indx) :: indx
character, intent(in), dimension(N_indx) :: opstr
complex(kind=4), intent(out), dimension(Ns) :: ME
integer(kind=4), intent(out), dimension(Ns) :: col
integer(kind=4), intent(out) :: error
integer(kind=4), external :: FindZstate
complex(kind=4) :: M_E
integer(kind=4) :: i,j,r,a

error=0
do i=1,Ns
M_E=1.0; r=basis(i)
do j=1,N_indx
a = iand(ishft(r,-indx(j)),1)
if(opstr(j) .eq. "c") then
cycle

else if(opstr(j) .eq. "z") then
M_E = M_E * (-1)**(a+1)*0.5

else if(opstr(j) .eq. "x") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * 0.5

else if(opstr(j) .eq. "y") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * (-1)**(a)*cmplx(0.0,0.5)

else if(opstr(j) .eq. "+") then
if (a .eq. 1) then 
col(i)=i; ME=0.0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else if(opstr(j) .eq. "-") then
if (a .eq. 0) then 
col(i)=i; ME=0.0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else
ME=0;col=0;error=1
return 

end if
end do
ME(i)=M_E
col(i)=r
end do



end subroutine





subroutine z_SpinOp(basis,Ns,opstr,indx,N_indx,ME,col,error)
implicit none
integer(kind=4), intent(in) :: Ns,N_indx
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=4), intent(in), dimension(N_indx) :: indx
character, intent(in), dimension(N_indx) :: opstr
complex(kind=8), intent(out), dimension(Ns) :: ME
integer(kind=4), intent(out), dimension(Ns) :: col
integer(kind=4), intent(out) :: error
integer(kind=4), external :: FindZstate
complex(kind=8) :: M_E
integer(kind=4) :: i,j,r,a

error=0
do i=1,Ns
M_E=1.0d0; r=basis(i)
do j=1,N_indx
a = iand(ishft(r,-indx(j)),1)
if(opstr(j) .eq. "c") then
cycle

else if(opstr(j) .eq. "z") then
M_E = M_E * (-1)**(a+1)*0.5d0

else if(opstr(j) .eq. "x") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * 0.5d0

else if(opstr(j) .eq. "y") then
r = ieor(r,ishft(1,indx(j)))
M_E = M_E * (-1)**(a)*dcmplx(0.0,0.5d0)

else if(opstr(j) .eq. "+") then
if (a .eq. 1) then 
col(i)=i; ME=0.0d0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else if(opstr(j) .eq. "-") then
if (a .eq. 0) then 
col(i)=i; ME=0.0d0;
exit
else
r = ieor(r,ishft(1,indx(j)))
end if

else
ME=0;col=0;error=1
return 

end if
end do
ME(i)=M_E
col(i)=r
end do



end subroutine














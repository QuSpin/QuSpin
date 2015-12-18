subroutine CheckState_T(s,L,k,a,r)
implicit none
integer(kind=4), intent(in) :: s,L,k,a
integer(kind=1), intent(out) :: r
integer(kind=4) :: t,i
t=s
do i=1,L,a
t=ishftc(t,a,L)
if ( t .lt. s ) then
r=-1
return

else if ( t .eq. s) then 

if ( modulo(k,(L/int(i))) .ne. 0) then
r=-1
return
end if

r=i
return

end if
end do

r=-1
return


end subroutine





subroutine make_t_basis(L,mbasis,Ns_m,kblock,a,N)
implicit none
integer(kind=4), intent(in) :: L, kblock, Ns_m,a
integer(kind=4), intent(inout) ,dimension(Ns_m) :: mbasis
integer(kind=1), intent(out), dimension(Ns_m) :: N
integer(kind=4) :: i
integer(kind=1) :: r


do i=1,Ns_m
call CheckState_T(mbasis(i),L,kblock,a,r)
N(i)=r
if(r .lt. 0) then
mbasis(i)=-1
end if
end do

end subroutine


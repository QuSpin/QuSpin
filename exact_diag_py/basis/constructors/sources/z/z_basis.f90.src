subroutine CheckState_Z(s,L,r)
implicit none
integer(kind=4), intent(in) :: s,L
integer(kind=1), intent(out) :: r
integer(kind=4) :: t
character(32), external :: bits
t=s 
call flip_all(t,L)
if(t .gt. s) then
r=2
else
r=-1
end if
end subroutine





subroutine make_z_basis(L,mbasis,Ns_m,N)
implicit none
integer(kind=4), intent(in) :: L, Ns_m
integer(kind=4), intent(inout) ,dimension(Ns_m) :: mbasis
integer(kind=1), intent(out), dimension(Ns_m) :: N
integer(kind=4) :: i
integer(kind=1) :: r

do i=1,Ns_m
call CheckState_Z(mbasis(i),L,r)
N(i)=r
if(r.lt.0) then
mbasis(i)=-1
end if
end do

end subroutine





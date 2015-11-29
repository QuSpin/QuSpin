subroutine CheckState_P(s,L,p,r)
implicit none
integer(kind=4), intent(in) :: s,L,p
integer(kind=1), intent(out) :: r
integer(kind=4) :: t
t=s
call flip_lr(t,L)
if(t .eq. s) then
if(p .ne. -1) then
r=4; return
else
r=-1; return
end if
else if(t .gt. s) then
r=2; return
else
r=-1; return
end if

end subroutine





subroutine make_p_basis(L,mbasis,Ns_m,pblock,N)
implicit none
integer(kind=4), intent(in) :: L, pblock, Ns_m
integer(kind=4), intent(inout) ,dimension(Ns_m) :: mbasis
integer(kind=1), intent(out), dimension(Ns_m) :: N
integer(kind=4) :: i
integer(kind=1) :: r


do i=1,Ns_m
call CheckState_P(mbasis(i),L,pblock,r)
N(i)=r
if(r .lt. 0) then
mbasis(i)=-1
end if
end do

end subroutine


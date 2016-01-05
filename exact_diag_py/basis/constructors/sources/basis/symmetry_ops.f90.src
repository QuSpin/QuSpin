!program test
!integer(kind=4) :: i
!i=3
!call flip_lr(i,3)
!print*, i
!end program




subroutine flip_lr(s,L)
implicit none
integer(kind=4), intent(inout) :: s
integer(kind=4), intent(in) :: L
integer(kind=4) :: upper, lower, s1, i
lower = 0
upper = L


s1=0
do i=0,L-1
if(btest(s,i)) then
s1=s1+ishft(1,L-1-i)
end if
end do
s=s1


end subroutine





subroutine flip_all(s,L)
implicit none
integer(kind=4), intent(inout) :: s
integer(kind=4), intent(in) :: L
integer(kind=4) :: upper, lower, i1, i2, i3
lower = 0
upper = L
i1 = ishft(1,upper)-1
i2 = not(ishft(1,lower)-1)
i3 = iand(i1,i2)
s=ieor(s,i3)
end subroutine 




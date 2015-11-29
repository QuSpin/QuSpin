subroutine RefState(basis,col,Ns)
implicit none
integer(kind=4), intent(in) :: Ns
integer(kind=4), intent(in), dimension(Ns) :: basis
integer(kind=4), intent(inout), dimension(Ns) :: col
integer(kind=4), external :: FindZstate
integer(kind=4) :: i

do i=1,Ns
col(i)=FindZstate(basis,Ns,col(i))
end do

end subroutine



function FindZstate(A,N,s)
implicit none
integer(kind=4) :: FindZstate
integer(kind=4), intent(in) :: N, s
integer(kind=4), intent(in), dimension(N) :: A
integer(kind=4) :: bmin,bmax,b

bmin=1;bmax=N
do while(bmin .le. bmax)
b=(bmin+bmax)/2
if(s < A(b)) then
bmax=b-1
else if(s > A(b)) then
bmin=b+1
else
FindZstate=b
return 
end if
end do
FindZstate=-1
end function

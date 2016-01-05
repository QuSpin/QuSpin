subroutine make_m_basis(s0,Ns,zbasis)
implicit none
Integer(kind=4), intent(in) :: s0,Ns
Integer(kind=4), intent(out), dimension(Ns) :: zbasis
Integer(kind=4) :: i,s,t


zbasis(1)=s0
s=s0
do i=2,Ns
t=ior(s,s-1)+1
s=ior(t,ishft(iand(t,-t)/iand(s,-s),-1)-1)
zbasis(i)=s
end do

end subroutine

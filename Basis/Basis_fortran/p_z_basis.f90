subroutine make_p_z_basis(L,mbasis,Ns_m,pblock,zblock,N)
implicit none
integer(kind=4), intent(in) :: L, pblock,zblock, Ns_m
integer(kind=4), intent(inout) ,dimension(Ns_m) :: mbasis
integer(kind=1), intent(out), dimension(Ns_m) :: N
integer(kind=4) :: i,s
integer(kind=1) :: rp,rz,rpz



do i=1,Ns_m
s=mbasis(i)
call CheckState_Z(s,L,rz)
if(rz.lt.0) then
mbasis(i)=-1
N(i)=-1
cycle
end if
call CheckState_P(s,L,pblock,rp)
if(rp.lt.0) then
mbasis(i)=-1
N(i)=-1
cycle
end if
call CheckState_PZ(s,L,pblock*zblock,rpz)
if(rpz.lt.0) then
mbasis(i)=-1
N(i)=-1
cycle
end if
N(i)=rp*rpz
end do


end subroutine













subroutine s_lanczos_op(m,col,nm,rowptr,nr,v_in,n,tol,ncv_max,a,b,ncv,stat)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv_max
real(kind=4), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
real(kind=4), intent(in), dimension(n) :: v_in
real(kind=4), intent(in) :: tol
real(kind=4), intent(out), dimension(ncv_max) :: a
real(kind=4), intent(out), dimension(ncv_max-1) :: b
integer(kind=4), intent(out) :: stat,ncv
! local variables
real(kind=4), dimension(n) :: q,r,w
integer(kind=4) :: j
real(kind=4) :: log_prod_b, log_fact, a_sum, log_tol

stat = -1
b = 0.0
a = 0.0

log_tol = log(tol)
log_fact = log(2.0)

q = v_in
call s_csr_dot(m,col,nm,rowptr,nr,q,r,n)

a(1) = real(dot_product(q,r))
r = r - a(1) * q
b(1) = sqrt(abs(dot_product(r,r)))

log_prod_b = log( b(1) )
a_sum = abs(a(1))
 
j = 2
do while((j .lt. ncv_max-1) .and. (stat .lt. 0))
w = q 
q = r/b(j-1)
call s_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = real(dot_product(q,r))
r = r - a(j) * q

b(j) = sqrt(abs(dot_product(r,r)))
log_fact = log_fact + log(real(j+1))
a_sum = a_sum + abs(a(j))

if( b(j) .gt. 0.0) then
log_prod_b = log_prod_b + log( b(j) )
else if( b(j) .eq. 0.0 )then
stat = 1
j = j + 1
exit
end if


j = j + 1
!print*,  log(a_sum) + log_prod_b - log_fact, log_tol
if( ( ( log(a_sum) + log_prod_b - log_fact) .lt. log_tol ) .and. (j .ge. 2) ) then
stat=0
exit
end if

end do

w = q 
q = r/b(j-1)
call s_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = real(dot_product(q,r))
ncv = j



 
end subroutine












subroutine d_lanczos_op(m,col,nm,rowptr,nr,v_in,n,tol,ncv_max,a,b,ncv,stat)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv_max
real(kind=8), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
real(kind=8), intent(in), dimension(n) :: v_in
real(kind=8), intent(in) :: tol
real(kind=8), intent(out), dimension(ncv_max) :: a
real(kind=8), intent(out), dimension(ncv_max-1) :: b
integer(kind=4), intent(out) :: stat,ncv
! local variables
real(kind=8), dimension(n) :: q,r,w
integer(kind=4) :: j
real(kind=8) :: log_prod_b, log_fact, a_sum, log_tol

stat = -1
b = 0.0d0
a = 0.0d0

log_tol = dlog(tol)
log_fact = dlog(2.0d0)

q = v_in
call d_csr_dot(m,col,nm,rowptr,nr,q,r,n)

a(1) = dble(dot_product(q,r))
r = r - a(1) * q
b(1) = dsqrt(dabs(dot_product(r,r)))

log_prod_b = dlog( b(1) )
a_sum = dabs(a(1))
 
j = 2
do while((j .lt. ncv_max-1) .and. (stat .lt. 0d0))
w = q 
q = r/b(j-1)
call d_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = dble(dot_product(q,r))
r = r - a(j) * q

b(j) = dsqrt(dabs(dot_product(r,r)))
log_fact = log_fact + dlog(dble(j+1))
a_sum = a_sum + dabs(a(j))

if( b(j) .gt. 0.0d0) then
log_prod_b = log_prod_b + dlog( b(j) )
else if( b(j) .eq. 0.0d0 )then
stat = 1
j = j + 1
exit
end if


j = j + 1
!print*,  dlog(a_sum) + log_prod_b - log_fact, log_tol
if( ( ( dlog(a_sum) + log_prod_b - log_fact) .lt. log_tol ) .and. (j .ge. 2) ) then
stat=0
exit
end if

end do

w = q 
q = r/b(j-1)
call d_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = dble(dot_product(q,r))
ncv = j



 
end subroutine












subroutine c_lanczos_op(m,col,nm,rowptr,nr,v_in,n,tol,ncv_max,a,b,ncv,stat)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv_max
complex(kind=4), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
complex(kind=4), intent(in), dimension(n) :: v_in
real(kind=4), intent(in) :: tol
real(kind=4), intent(out), dimension(ncv_max) :: a
real(kind=4), intent(out), dimension(ncv_max-1) :: b
integer(kind=4), intent(out) :: stat,ncv
! local variables
complex(kind=4), dimension(n) :: q,r,w
integer(kind=4) :: j
real(kind=4) :: log_prod_b, log_fact, a_sum, log_tol

stat = -1
b = 0.0
a = 0.0

log_tol = log(tol)
log_fact = log(2.0)

q = v_in
call c_csr_dot(m,col,nm,rowptr,nr,q,r,n)

a(1) = real(dot_product(q,r))
r = r - a(1) * q
b(1) = sqrt(cabs(dot_product(r,r)))

log_prod_b = log( b(1) )
a_sum = abs(a(1))
 
j = 2
do while((j .lt. ncv_max-1) .and. (stat .lt. 0))
w = q 
q = r/b(j-1)
call c_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = real(dot_product(q,r))
r = r - a(j) * q

b(j) = sqrt(cabs(dot_product(r,r)))
log_fact = log_fact + log(real(j+1))
a_sum = a_sum + abs(a(j))

if( b(j) .gt. 0.0) then
log_prod_b = log_prod_b + log( b(j) )
else if( b(j) .eq. 0.0 )then
stat = 1
j = j + 1
exit
end if


j = j + 1
!print*,  log(a_sum) + log_prod_b - log_fact, log_tol
if( ( ( log(a_sum) + log_prod_b - log_fact) .lt. log_tol ) .and. (j .ge. 2) ) then
stat=0
exit
end if

end do

w = q 
q = r/b(j-1)
call c_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = real(dot_product(q,r))
ncv = j



 
end subroutine












subroutine z_lanczos_op(m,col,nm,rowptr,nr,v_in,n,tol,ncv_max,a,b,ncv,stat)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv_max
complex(kind=8), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
complex(kind=8), intent(in), dimension(n) :: v_in
real(kind=8), intent(in) :: tol
real(kind=8), intent(out), dimension(ncv_max) :: a
real(kind=8), intent(out), dimension(ncv_max-1) :: b
integer(kind=4), intent(out) :: stat,ncv
! local variables
complex(kind=8), dimension(n) :: q,r,w
integer(kind=4) :: j
real(kind=8) :: log_prod_b, log_fact, a_sum, log_tol

stat = -1
b = 0.0d0
a = 0.0d0

log_tol = dlog(tol)
log_fact = dlog(2.0d0)

q = v_in
call z_csr_dot(m,col,nm,rowptr,nr,q,r,n)

a(1) = dble(dot_product(q,r))
r = r - a(1) * q
b(1) = dsqrt(zabs(dot_product(r,r)))

log_prod_b = dlog( b(1) )
a_sum = dabs(a(1))
 
j = 2
do while((j .lt. ncv_max-1) .and. (stat .lt. 0d0))
w = q 
q = r/b(j-1)
call z_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = dble(dot_product(q,r))
r = r - a(j) * q

b(j) = dsqrt(zabs(dot_product(r,r)))
log_fact = log_fact + dlog(dble(j+1))
a_sum = a_sum + dabs(a(j))

if( b(j) .gt. 0.0d0) then
log_prod_b = log_prod_b + dlog( b(j) )
else if( b(j) .eq. 0.0d0 )then
stat = 1
j = j + 1
exit
end if


j = j + 1
!print*,  dlog(a_sum) + log_prod_b - log_fact, log_tol
if( ( ( dlog(a_sum) + log_prod_b - log_fact) .lt. log_tol ) .and. (j .ge. 2) ) then
stat=0
exit
end if

end do

w = q 
q = r/b(j-1)
call z_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w
a(j) = dble(dot_product(q,r))
ncv = j



 
end subroutine
















subroutine s_get_vec(m,col,nm,rowptr,nr,v_l,a,b,ncv,v_in,n,v_out)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv
real(kind=4), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
real(kind=4), intent(in), dimension(n) :: v_in
real(kind=4), intent(in), dimension(ncv) :: v_l
real(kind=4), intent(in), dimension(ncv) :: a
real(kind=4), intent(in), dimension(ncv-1) :: b
real(kind=4), intent(out), dimension(n) :: v_out
! local variables
real(kind=4), dimension(n) :: q,r,w
integer(kind=4) :: j

q = v_in

v_out = v_out + v_l(1) * q
call s_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - a(1) * q

do j = 2, ncv - 1
w = q 
q = r/b(j-1)
v_out = v_out + v_l(j) * q
call s_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w - a(j) * q
end do
 
w = q 
q = r/b(ncv-1)
v_out = v_out + v_l(ncv) * q
 
end subroutine















subroutine d_get_vec(m,col,nm,rowptr,nr,v_l,a,b,ncv,v_in,n,v_out)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv
real(kind=8), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
real(kind=8), intent(in), dimension(n) :: v_in
real(kind=8), intent(in), dimension(ncv) :: v_l
real(kind=8), intent(in), dimension(ncv) :: a
real(kind=8), intent(in), dimension(ncv-1) :: b
real(kind=8), intent(out), dimension(n) :: v_out
! local variables
real(kind=8), dimension(n) :: q,r,w
integer(kind=4) :: j

q = v_in

v_out = v_out + v_l(1) * q
call d_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - a(1) * q

do j = 2, ncv - 1
w = q 
q = r/b(j-1)
v_out = v_out + v_l(j) * q
call d_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w - a(j) * q
end do
 
w = q 
q = r/b(ncv-1)
v_out = v_out + v_l(ncv) * q
 
end subroutine















subroutine c_get_vec(m,col,nm,rowptr,nr,v_l,a,b,ncv,v_in,n,v_out)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv
complex(kind=4), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
complex(kind=4), intent(in), dimension(n) :: v_in
complex(kind=4), intent(in), dimension(ncv) :: v_l
real(kind=4), intent(in), dimension(ncv) :: a
real(kind=4), intent(in), dimension(ncv-1) :: b
complex(kind=4), intent(out), dimension(n) :: v_out
! local variables
complex(kind=4), dimension(n) :: q,r,w
integer(kind=4) :: j

q = v_in

v_out = v_out + v_l(1) * q
call c_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - a(1) * q

do j = 2, ncv - 1
w = q 
q = r/b(j-1)
v_out = v_out + v_l(j) * q
call c_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w - a(j) * q
end do
 
w = q 
q = r/b(ncv-1)
v_out = v_out + v_l(ncv) * q
 
end subroutine















subroutine z_get_vec(m,col,nm,rowptr,nr,v_l,a,b,ncv,v_in,n,v_out)
implicit none
integer(kind=4), intent(in) :: nm,nr,n,ncv
complex(kind=8), intent(in), dimension(nm) :: m
integer(kind=4), intent(in), dimension(nm) :: col
integer(kind=4), intent(in), dimension(nr) :: rowptr
complex(kind=8), intent(in), dimension(n) :: v_in
complex(kind=8), intent(in), dimension(ncv) :: v_l
real(kind=8), intent(in), dimension(ncv) :: a
real(kind=8), intent(in), dimension(ncv-1) :: b
complex(kind=8), intent(out), dimension(n) :: v_out
! local variables
complex(kind=8), dimension(n) :: q,r,w
integer(kind=4) :: j

q = v_in

v_out = v_out + v_l(1) * q
call z_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - a(1) * q

do j = 2, ncv - 1
w = q 
q = r/b(j-1)
v_out = v_out + v_l(j) * q
call z_csr_dot(m,col,nm,rowptr,nr,q,r,n)
r = r - b(j-1) * w - a(j) * q
end do
 
w = q 
q = r/b(ncv-1)
v_out = v_out + v_l(ncv) * q
 
end subroutine











subroutine s_csr_dot(a,col,na,rowptr,nr,v_in,v_out,n)
implicit none
integer(kind=4), intent(in) :: na,nr,n
real(kind=4), intent(in), dimension(0:na-1) :: a
integer(kind=4), intent(in), dimension(0:na-1) :: col
integer(kind=4), intent(in), dimension(0:nr-1) :: rowptr
real(kind=4), intent(in), dimension(0:n-1) :: v_in
real(kind=4), intent(out), dimension(0:n-1) :: v_out
integer(kind=4) :: i,j
 
v_out=0.0
 
do i = 0, n - 1
do j = rowptr(i), rowptr(i+1) - 1
v_out(i) = v_out(i) + A(j) * v_in(col(j))
end do
end do
 
end subroutine










subroutine d_csr_dot(a,col,na,rowptr,nr,v_in,v_out,n)
implicit none
integer(kind=4), intent(in) :: na,nr,n
real(kind=8), intent(in), dimension(0:na-1) :: a
integer(kind=4), intent(in), dimension(0:na-1) :: col
integer(kind=4), intent(in), dimension(0:nr-1) :: rowptr
real(kind=8), intent(in), dimension(0:n-1) :: v_in
real(kind=8), intent(out), dimension(0:n-1) :: v_out
integer(kind=4) :: i,j
 
v_out=0.0
 
do i = 0, n - 1
do j = rowptr(i), rowptr(i+1) - 1
v_out(i) = v_out(i) + A(j) * v_in(col(j))
end do
end do
 
end subroutine










subroutine c_csr_dot(a,col,na,rowptr,nr,v_in,v_out,n)
implicit none
integer(kind=4), intent(in) :: na,nr,n
complex(kind=4), intent(in), dimension(0:na-1) :: a
integer(kind=4), intent(in), dimension(0:na-1) :: col
integer(kind=4), intent(in), dimension(0:nr-1) :: rowptr
complex(kind=4), intent(in), dimension(0:n-1) :: v_in
complex(kind=4), intent(out), dimension(0:n-1) :: v_out
integer(kind=4) :: i,j
 
v_out=0.0
 
do i = 0, n - 1
do j = rowptr(i), rowptr(i+1) - 1
v_out(i) = v_out(i) + A(j) * v_in(col(j))
end do
end do
 
end subroutine










subroutine z_csr_dot(a,col,na,rowptr,nr,v_in,v_out,n)
implicit none
integer(kind=4), intent(in) :: na,nr,n
complex(kind=8), intent(in), dimension(0:na-1) :: a
integer(kind=4), intent(in), dimension(0:na-1) :: col
integer(kind=4), intent(in), dimension(0:nr-1) :: rowptr
complex(kind=8), intent(in), dimension(0:n-1) :: v_in
complex(kind=8), intent(out), dimension(0:n-1) :: v_out
integer(kind=4) :: i,j
 
v_out=0.0
 
do i = 0, n - 1
do j = rowptr(i), rowptr(i+1) - 1
v_out(i) = v_out(i) + A(j) * v_in(col(j))
end do
end do
 
end subroutine
















!subroutine <name>_get_basis(m,col,nm,rowptr,nr,a,b,ncv,v_in,n,v)
!implicit none
!integer(kind=4), intent(in) :: nm,nr,n,ncv
!<type>, intent(in), dimension(nm) :: m
!integer(kind=4), intent(in), dimension(nm) :: col
!integer(kind=4), intent(in), dimension(nr) :: rowptr
!<type>, intent(in), dimension(n) :: v_in
!<abstype>, intent(in), dimension(ncv) :: a
!<abstype>, intent(in), dimension(ncv-1) :: b
!<type>, intent(out), dimension(n,ncv) :: v
! local variables
!<type>, dimension(n) :: q,r,w
!integer(kind=4) :: j
!v(:,1) = v_in
!q = v_in
!call <name>_csr_dot(m,col,nm,rowptr,nr,q,r,n)
!r = r - a(1) * q
!do j = 2, ncv - 1
!w = q 
!q = r/b(j-1)
!v(:,j) = q
!call <name>_csr_dot(m,col,nm,rowptr,nr,q,r,n)
!r = r - b(j-1) * w - a(j) * q
!end do
!w = q 
!q = r/b(ncv-1)
!v(:,ncv) = q
!end subroutine









!subroutine <name>_full_orthogonal(v,n,p,r,w)
!implicit none
!integer(kind=4), intent(in) :: n,p
!<type>, intent(in), dimension(n,p) :: v
!<type>, intent(in), dimension(n) :: r
!<type>, intent(out), dimension(n) :: w
!<type>, dimension(p) :: rw
!rw = matmul(transpose(<v,v,conjg(v),dconjg(v)>),r)
!w = matmul(v,rw)
!end subroutine







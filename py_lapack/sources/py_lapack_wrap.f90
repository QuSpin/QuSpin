subroutine py_dsyev(JOBZ,UPLO,N,A,W,INFO)
implicit none
character(1), intent(in) :: JOBZ,UPLO
integer(kind=4), intent(in) :: N
real(kind=8), intent(inout), dimension(N,N) :: A
real(kind=8), intent(out), dimension(N) :: W
integer(kind=4), intent(out) :: INFO
integer(kind=4) :: LWORK
real(kind=8), allocatable, dimension(:) :: WORK

allocate(WORK(1))
call dsyev(JOBZ,UPLO,N,A,N,W,WORK,-1,INFO)
if(info.eq.0) then
LWORK=int(WORK(1))
if(LWORK.le.0) then
INFO=-N-1
return
end if
deallocate(WORK)
allocate(WORK(LWORK))
else
return
end if

call dsyev(JOBZ,UPLO,N,A,N,W,WORK,LWORK,INFO)

end subroutine



subroutine py_ssyev(JOBZ,UPLO,N,A,W,INFO)
implicit none
character(1), intent(in) :: JOBZ,UPLO
integer(kind=4), intent(in) :: N
real(kind=4), intent(inout), dimension(N,N) :: A
real(kind=4), intent(out), dimension(N) :: W
integer(kind=4), intent(out) :: INFO
integer(kind=4) :: LWORK
real(kind=4), allocatable, dimension(:) :: WORK

allocate(WORK(1))
call ssyev(JOBZ,UPLO,N,A,N,W,WORK,-1,INFO)
if(info.eq.0) then
LWORK=int(WORK(1))
if(LWORK.le.0) then
INFO=-N-1
return
end if
deallocate(WORK)
allocate(WORK(LWORK))
else
return
end if

call ssyev(JOBZ,UPLO,N,A,N,W,WORK,LWORK,INFO)

end subroutine




subroutine py_cheev(JOBZ,UPLO,N,A,W,INFO)
implicit none
character(1), intent(in) :: JOBZ,UPLO
integer(kind=4), intent(in) :: N
complex(kind=4), intent(inout), dimension(N,N) :: A
real(kind=4), intent(out), dimension(N) :: W
integer(kind=4), intent(out) :: INFO
integer(kind=4) :: LWORK
complex(kind=4), allocatable, dimension(:) :: WORK
real(kind=4), dimension(3*N-2) :: RWORK

allocate(WORK(1))
call cheev(JOBZ,UPLO,N,A,N,W,WORK,-1,RWORK,INFO)
if(info.eq.0) then
LWORK=int(WORK(1))
if(LWORK.le.0) then
INFO=-N-1
return
end if
deallocate(WORK)
allocate(WORK(LWORK))
else
return
end if

call cheev(JOBZ,UPLO,N,A,N,W,WORK,LWORK,RWORK,INFO)

end subroutine







subroutine py_zheev(JOBZ,UPLO,N,A,W,INFO)
implicit none
character(1), intent(in) :: JOBZ,UPLO
integer(kind=4), intent(in) :: N
complex(kind=8), intent(inout), dimension(N,N) :: A
real(kind=8), intent(out), dimension(N) :: W
integer(kind=4), intent(out) :: INFO
integer(kind=4) :: LWORK
complex(kind=8), allocatable, dimension(:) :: WORK
real(kind=8), dimension(3*N-2) :: RWORK

allocate(WORK(1))
call zheev(JOBZ,UPLO,N,A,N,W,WORK,-1,RWORK,INFO)
if(info.eq.0) then
LWORK=int(WORK(1))
if(LWORK.le.0) then
INFO=-N-1
return
end if
deallocate(WORK)
allocate(WORK(LWORK))
else
return
end if

call zheev(JOBZ,UPLO,N,A,N,W,WORK,LWORK,RWORK,INFO)

end subroutine









subroutine py_cgeev(JOBL,JOBR,N,A,W,INFO)
implicit none
character(1), intent(in) :: JOBL,JOBR
integer(kind=4), intent(in) :: N
complex(kind=4), intent(inout), dimension(N,N) :: A
complex(kind=4), intent(out), dimension(N) :: W
integer(kind=4), intent(out) :: INFO
real(kind=4), dimension(2*N) :: RWORK
complex(kind=4), allocatable, dimension(:,:) :: VL,VR
complex(kind=4), allocatable, dimension(:) :: WORK
integer(kind=4) :: LWORK, LDVL, LDVR



if(JOBR.eq.'V'.and. JOBL.eq.'V') then
info=2790
return
end if

if(JOBL.eq.'V') then
LDVL=N; allocate(VL(N,LDVL))
else
LDVL=1; allocate(VL(1,1))
end if

if(JOBR.eq.'V') then
LDVR=N; allocate(VR(N,LDVR))
else
LDVR=1; allocate(VR(1,1))
end if




allocate(WORK(1))
call cgeev(JOBL,JOBR,N,A,N,W,VL,LDVL,VR,LDVR,WORK,-1,RWORK,INFO)
if(info.eq.0) then
LWORK=int(WORK(1))
if(LWORK.le.0) then
INFO=-N-1
return
end if
deallocate(WORK)
allocate(WORK(LWORK))
else
return
end if

call cgeev(JOBL,JOBR,N,A,N,W,VL,LDVL,VR,LDVR,WORK,LWORK,RWORK,INFO)

if(JOBL.eq.'V') then
A=VL
else if(JOBR.eq.'V') then
A=VR
end if

end subroutine









subroutine py_zgeev(JOBL,JOBR,N,A,W,INFO)
implicit none
character(1), intent(in) :: JOBL,JOBR
integer(kind=4), intent(in) :: N
complex(kind=8), intent(inout), dimension(N,N) :: A
complex(kind=8), intent(out), dimension(N) :: W
integer(kind=4), intent(out) :: INFO
real(kind=8), dimension(2*N) :: RWORK
complex(kind=8), allocatable, dimension(:,:) :: VL,VR
complex(kind=8), allocatable, dimension(:) :: WORK
integer(kind=4) :: LWORK, LDVL, LDVR


if(JOBR.eq.'V'.and. JOBL.eq.'V') then
info=2790
return
end if

if(JOBL.eq.'V') then
LDVL=N; allocate(VL(N,LDVL))
else
LDVL=1; allocate(VL(1,1))
end if

if(JOBR.eq.'V') then
LDVR=N; allocate(VR(N,LDVR))
else
LDVR=1; allocate(VR(1,1))
end if




allocate(WORK(1))
call zgeev(JOBL,JOBR,N,A,N,W,VL,LDVL,VR,LDVR,WORK,-1,RWORK,INFO)
if(info.eq.0) then
LWORK=int(WORK(1))
if(LWORK.le.0) then
INFO=-N-1
return
end if
deallocate(WORK)
allocate(WORK(LWORK))
else
return
end if

call zgeev(JOBL,JOBR,N,A,N,W,VL,LDVL,VR,LDVR,WORK,LWORK,RWORK,INFO)

if(JOBL.eq.'V') then
A=VL
else if(JOBR.eq.'V') then
A=VR
end if

end subroutine







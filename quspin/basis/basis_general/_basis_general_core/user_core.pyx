# cython: language_level=2
# distutils: language=c++
import cython
cimport numpy as _np
import numpy as _np
from numba.ccallback import CFunc
from libcpp.vector cimport vector


include "source/general_basis_core.pyx"


# specialized code 
cdef extern from "user_basis_core.h" namespace "basis_general":
    cdef cppclass user_basis_core[I](general_basis_core[I]):
        user_basis_core(const int,const int,void *, 
            const int*, const int*,I **, const int,size_t,I*,size_t,I*,size_t,size_t)

cdef class user_core_wrap(general_basis_core_wrap):
    cdef object get_Ns_pcon_py
    cdef object get_s0_pcon_py
    cdef object next_state
    cdef object pre_check_state
    cdef object count_particles
    cdef object op_func
    cdef _np.ndarray maps_arr 
    cdef _np.ndarray pers_arr
    cdef _np.ndarray qs_arr
    cdef _np.ndarray maps_args_arr
    cdef _np.ndarray ns_args_arr
    cdef _np.ndarray precs_args_arr
    cdef int n_sectors
    cdef vector[void*] maps_args_vector 
        

    def __cinit__(self,Ns_full,dtype,N,maps, pers, qs, maps_args,
        int n_sectors, get_Ns_pcon, get_s0_pcon, next_state,
        ns_args,pre_check_state,precs_args,count_particles, op_func, sps):
        self._N = N
        self._Ns_full = Ns_full
        self.get_s0_pcon_py = get_s0_pcon
        self.get_Ns_pcon_py = get_Ns_pcon
        self.next_state = next_state
        self.pre_check_state = pre_check_state
        self.count_particles = count_particles
        self.op_func = op_func
        self.n_sectors = n_sectors

        if sps is None:
            sps = -1


        cdef size_t next_state_add=0,pre_check_state_add=0,count_particles_add=0,op_func_add=0

        if next_state is not None:
            if not isinstance(next_state,CFunc):
                raise ValueError("next_state must be a numba CFunc object.")

            if get_Ns_pcon is None or get_Ns_pcon is None:
                raise RuntimeError("user must define get_Ns_pcon and get_s0_pcon functions to use particle conservation.")

            next_state_add = next_state.address

        if pre_check_state is not None:
            if not isinstance(pre_check_state,CFunc):
                raise ValueError("pre_check_state must be a numba CFunc object.")

            pre_check_state_add = pre_check_state.address

        if count_particles is not None:
            if not isinstance(count_particles,CFunc):
                raise ValueError("next_state must be a numba CFunc object.")

            count_particles_add = count_particles.address

        if op_func is not None:
            if not isinstance(op_func,CFunc):
                raise ValueError("next_state must be a numba CFunc object.")

            op_func_add = op_func.address
        else:
            raise RuntimeError("user basis requires at least an op_func to do any calculations.")

        maps_addresses = []
        for map_func in maps:
            if not isinstance(map_func,CFunc):
                raise ValueError("map functions must be a numba Cfunc object.")

            maps_addresses.append(map_func.address)


        self.maps_arr = _np.array(maps_addresses)
        self.pers_arr = _np.array(pers,dtype=_np.intc)
        self.qs_arr   = _np.array(qs,dtype=_np.intc)
        self.ns_args_arr = ns_args
        self.precs_args_arr = precs_args
        self._nt = self.maps_arr.shape[0]
        cdef void * maps_ptr = NULL
        cdef void * ns_args_ptr = NULL
        cdef void * precs_args_ptr = NULL
        cdef int  * pers_ptr = NULL
        cdef int  * qs_ptr   = NULL

        if self._nt > 0:
            maps_ptr = _np.PyArray_DATA(self.maps_arr)
            pers_ptr = <int*>_np.PyArray_DATA(self.pers_arr)
            qs_ptr   = <int*>_np.PyArray_DATA(self.qs_arr)

            if maps_args is not None:
                for arg in maps_args:
                   self.maps_args_vector.push_back(_np.PyArray_DATA(arg))

        if ns_args is not None:
            ns_args_ptr = _np.PyArray_DATA(self.ns_args_arr)

        if precs_args is not None:
            precs_args_ptr = _np.PyArray_DATA(self.precs_args_arr)

        if dtype == uint32:
            self._basis_core = <void *> new user_basis_core[uint32_t](N,self._nt,maps_ptr,pers_ptr,qs_ptr,<uint32_t**>&self.maps_args_vector[0],
                n_sectors,next_state_add,<uint32_t*>ns_args_ptr,pre_check_state_add,<uint32_t*>precs_args_ptr,
                count_particles_add,op_func_add)
        elif dtype == uint64:
            self._basis_core = <void *> new user_basis_core[uint64_t](N,self._nt,maps_ptr,pers_ptr,qs_ptr,<uint64_t**>&self.maps_args_vector[0],
                n_sectors,next_state_add,<uint64_t*>ns_args_ptr,pre_check_state_add,<uint64_t*>precs_args_ptr,
                count_particles_add,op_func_add)
        else:
            raise ValueError("user defined basis only supports system sizes <= 64.")

    def get_s0_pcon(self,object Np):
        return self.get_s0_pcon_py(self._N,Np)

    def get_Ns_pcon(self,object Np):
        return self.get_Ns_pcon_py(self._N,Np)

    @cython.boundscheck(False)
    def make_basis(self,_np.ndarray basis,norm_type[:] n,object Np=None,uint8_t[:] count=None):
        cdef int Ns_1 = 0
        cdef int Ns_2 = 0
        cdef int Ns_3 = 0
        cdef npy_intp i = 0
        cdef mem_MAX = basis.shape[0]

        if Np is not None and self.next_state is None:
            raise RuntimeError("attemping to create particle conserving basis with no way of generating particle states. Define next_state cfunc (see doc).")

        if Np is None:
            Ns_2 = general_basis_core_wrap._make_basis_full(self,basis,n)
        elif type(Np) is tuple or type(Np) is int:
            Ns_2 = general_basis_core_wrap._make_basis_pcon(self,Np,basis,n)
        else:
            Np_iter = iter(Np)
            if count is None:
                for np in Np_iter:
                    Ns_1 = general_basis_core_wrap._make_basis_pcon(self,np,basis[Ns_2:],n[Ns_2:])
                    if Ns_1 < 0:
                        return Ns_1
                    else:
                        Ns_2 += Ns_1

                    if Ns_2 > mem_MAX:
                        return -1
            else:

                for np in Np_iter:
                    Ns_1 = general_basis_core_wrap._make_basis_pcon(self,np,basis[Ns_2:],n[Ns_2:])
                    if Ns_1 < 0:
                        return Ns_1
                    else:
                        Ns_3 = Ns_2 + Ns_1
                        for i in range(Ns_2,Ns_3,1):
                            count[i] = np

                        Ns_2 = Ns_3

                    if Ns_2 > mem_MAX:
                        return -1

        return Ns_2


    def op_bra_ket(self,*args):
        if self.count_particles is None:
            raise RuntimeError("op_bra_ket features requires the user to define a 'count_particles' cfunc and 'n_sectors' argument.")

        return general_basis_core_wrap.op_bra_ket(self,*args)


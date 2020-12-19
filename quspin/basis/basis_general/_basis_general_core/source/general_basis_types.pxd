# distutils: language=c++
from numpy cimport npy_intp,int8_t,int16_t,int32_t,int64_t,uint8_t,uint16_t,uint32_t,uint64_t
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from libcpp cimport bool


cdef extern from "bits_info.h" namespace "basis_general":
    cdef cppclass uint256_t:
        uint256_t operator&(int)
        uint256_t operator>>(int)
        uint256_t operator<<(int)
        uint256_t operator^(uint256_t)
        bool operator==(uint256_t)
        bool operator!=(uint256_t)
        bool operator!=(int)

    cdef cppclass uint1024_t:
        uint1024_t operator&(int)
        uint1024_t operator>>(int)
        uint1024_t operator<<(int)
        uint1024_t operator^(uint1024_t)
        uint1024_t& operator[](int)
        bool operator==(uint1024_t)
        bool operator!=(uint1024_t)
        bool operator!=(int)

    cdef cppclass uint4096_t:
        uint4096_t operator&(int)
        uint4096_t operator>>(int)
        uint4096_t operator<<(int)
        uint4096_t operator^(uint4096_t)
        bool operator==(uint4096_t)
        bool operator!=(uint4096_t)
        bool operator!=(int)

    cdef cppclass uint16384_t:
        uint16384_t operator&(int)
        uint16384_t operator>>(int)
        uint16384_t operator<<(int)
        uint16384_t operator^(uint16384_t)
        bool operator==(uint16384_t)
        bool operator!=(uint16384_t)
        bool operator!=(int)

ctypedef fused index_type:
    int32_t
    int64_t

ctypedef fused dtype:
    float32_t
    float64_t
    complex64_t
    complex128_t

ctypedef fused norm_type:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused norm_type_2:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused shift_type:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused npy_uint:
    uint32_t
    uint64_t    

ctypedef fused state_type:
    uint32_t
    uint64_t
    uint256_t
    uint1024_t
    uint4096_t
    uint16384_t
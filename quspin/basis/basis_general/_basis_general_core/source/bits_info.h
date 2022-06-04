#ifndef __BITS_INFO_H__
#define __BITS_INFO_H__

#include "numpy/ndarraytypes.h"
#include "boost/multiprecision/cpp_int.hpp"
#include "boost/numeric/conversion/cast.hpp"

namespace basis_general {

typedef boost::multiprecision::uint128_t uint128_t;
typedef boost::multiprecision::uint256_t uint256_t;
typedef boost::multiprecision::uint512_t uint512_t;
typedef boost::multiprecision::uint1024_t uint1024_t;

typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<2048, 2048, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void> > uint2048_t;
typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<4096, 4096, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void> > uint4096_t;
typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<8192, 8192, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void> > uint8192_t;
typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<16384, 16384, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void> > uint16384_t;

template<class J,class I>
inline J integer_cast(I s){
  try 
  {
    return boost::numeric_cast<J>(s);; // This conversion succeeds (is in range)
  }
  catch(boost::numeric::positive_overflow& e) {
    return -1;
  }
}

template<typename I>
struct bit_info{};

template<>
struct bit_info<uint16384_t>
{ enum {ld_bits=14,bits=16384,bytes=2048};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint8192_t>
{ enum {ld_bits=13,bits=4096,bytes=1024};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint4096_t>
{ enum {ld_bits=12,bits=4096,bytes=512};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint2048_t>
{ enum {ld_bits=11,bits=2048,bytes=256};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint1024_t>
{ enum {ld_bits=10,bits=1024,bytes=128};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint512_t>
{ enum {ld_bits=9,bits=512,bytes=64};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint256_t>
{ enum {ld_bits=8,bits=256,bytes=32};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint128_t>
{ enum {ld_bits=7,bits=128,bytes=16};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint64>
{  enum {ld_bits=6,bits=64,bytes=8};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint32>
{  enum {ld_bits=5,bits=32,bytes=4};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint16>
{  enum {ld_bits=4,bits=16,bytes=2};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint8>
{ enum {ld_bits=3,bits=8,bytes=1};
  typedef int bit_index_type;
};


template<class T>
typename bit_info<T>::bit_index_type bit_pos(T x, typename bit_info<T>::bit_index_type *idx)
{
  typename bit_info<T>::bit_index_type n = 0;
  do {
    if (x & 1) *(idx++) = n;
    n++;
  } while (x >>= 1); 
  return n;
}


template<class T>
int inline bit_count(T v,int l){
  v = v & (((~(T)0) >> 1) >> (bit_info<T>::bits - 1 - l));
  v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
  v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
  v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
  T res = (T)(v * ((T)~(T)0/255)) >> ((bit_info<T>::bytes - 1) * 8); // count
  return (int)res;

}


}

#endif
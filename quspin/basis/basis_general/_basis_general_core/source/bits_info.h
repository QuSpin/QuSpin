#ifndef __BITS_INFO_H__
#define __BITS_INFO_H__

#include "numpy/ndarraytypes.h"

#include "boost/multiprecision/cpp_int.hpp"
typedef boost::multiprecision::uint128_t uint128_t;
typedef boost::multiprecision::uint256_t uint256_t;
typedef boost::multiprecision::uint512_t uint512_t;
typedef boost::multiprecision::uint1024_t uint1024_t;

template<typename I>
struct bit_info{};

template<>
struct bit_info<uint1024_t>
{ enum {ld_bits=10,bits=1024};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint512_t>
{ enum {ld_bits=9,bits=512};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint256_t>
{ enum {ld_bits=8,bits=256};
  typedef int bit_index_type;
};

template<>
struct bit_info<uint128_t>
{ enum {ld_bits=7,bits=128};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint64>
{	enum {ld_bits=6,bits=64,bytes=8};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint32>
{	enum {ld_bits=5,bits=32,bytes=4};
  typedef int bit_index_type;
};

template<>
struct bit_info<npy_uint16>
{	enum {ld_bits=4,bits=16,bytes=2};
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
T inline bit_count(T v,int l){
  v = v & (((~(T)0) >> 1) >> (bit_info<T>::bits - 1 - l));
  v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
  v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
  v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
  return (T)(v * ((T)~(T)0/255)) >> ((bit_info<T>::bytes - 1) * 8); // count

}




#endif
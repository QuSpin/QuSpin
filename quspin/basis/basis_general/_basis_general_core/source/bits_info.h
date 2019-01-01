#ifndef __BITS_INFO_H__
#define __BITS_INFO_H__

#include "numpy/ndarraytypes.h"
typedef npy_uint8 uint8_t;
typedef npy_uint16 uint16_t;
typedef npy_uint32 uint32_t;
typedef npy_uint64 uint64_t;

// #include "boost/multiprecision/cpp_int.hpp"
// typedef boost::multiprecision::uint128_t uint128_t;
// typedef boost::multiprecision::uint256_t uint256_t;
// typedef boost::multiprecision::uint512_t uint512_t;
// typedef boost::multiprecision::uint1024_t uint1024_t;

template<typename T,int N>
struct eob  // every other bit 1
{enum{val=((T)eob<T,N-1>::val << (T)2)+(T)1};};

template<typename T>
struct eob<T,1>
{enum{val=(T)1};};

template<typename I>
struct bit_info
{};

// template<>
// struct bit_info<uint128_t>
// { enum {ld_bits=7,bits=128,
//   eob=eob<uint128_t,64>::val,all_bits=~(uint128_t)0};};

// template<>
// struct bit_info<uint256_t>
// { enum {ld_bits=8,bits=256,
//   eob=eob<uint256_t,128>::val,all_bits=~(uint256_t)0};};

// template<>
// struct bit_info<uint512_t>
// { enum {ld_bits=9,bits=512,
//   eob=eob<uint512_t,256>::val,all_bits=~(uint512_t)0};};

// template<>
// struct bit_info<uint1024_t>
// { enum {ld_bits=10,bits=1024,
//   eob=eob<uint1024_t,512>::val,all_bits=~(uint1024_t)0};};

template<>
struct bit_info<uint64_t>
{	enum {ld_bits=6,bits=64,bytes=8,
  eob=eob<uint64_t,32>::val,all_bits=~(uint64_t)0};};

template<>
struct bit_info<uint32_t>
{	enum {ld_bits=5,bits=32,bytes=4,
  eob=eob<uint32_t,16>::val,all_bits=~(uint32_t)0};};

template<>
struct bit_info<uint16_t>
{	enum {ld_bits=4,bits=16,bytes=2,
  eob=eob<uint16_t,8>::val,all_bits=~(uint16_t)0};};

template<>
struct bit_info<uint8_t>
{ enum {ld_bits=3,bits=8,bytes=1,
  eob=eob<uint8_t,4>::val,all_bits=~(uint8_t)0};};




#if defined(__GNUC__) || defined(__GNUG__)

template<class T>
T inline bit_count(T v,int l){
  v = v & ((bit_info<T>::all_bits >> 1) >> (bit_info<T>::bits - 1 - l));

  if(std::is_same<T, uint64_t>::value){
    return __builtin_popcountll(v);
  }
  else if(std::is_same<T, uint32_t>::value){
    return __builtin_popcount(v);
  }
  else if(std::is_same<T, uint16_t>::value){
    return __builtin_popcount((uint32_t)v);
  }
  else if(std::is_same<T, uint8_t>::value){
    return __builtin_popcount((uint32_t)v);
  }
  else{
    v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
    v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
    return (T)(v * ((T)~(T)0/255)) >> ((bit_info<T>::bytes - 1) * 8); // count
  }

}
#else

template<class T>
T inline bit_count(T v,int l){
  v = v & ((bit_info<T>::all_bits >> 1) >> (bit_info<T>::bits - 1 - l));
  v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
  v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
  v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
  return (T)(v * ((T)~(T)0/255)) >> ((bit_info<T>::bytes - 1) * 8); // count

}

#endif




#endif
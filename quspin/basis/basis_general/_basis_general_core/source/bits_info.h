#ifndef __BITS_INFO_H__
#define __BITS_INFO_H__

#include "numpy/ndarraytypes.h"


template<typename I>
struct bit_info
{};

template<>
struct bit_info<npy_uint64>
{	enum {ld_bits=6,bits=64,
        eob=0x5555555555555555ull, // every other bit 1
        all_bits=0xffffffffffffffffull};};

template<>
struct bit_info<npy_uint32>
{	enum {ld_bits=5,bits=32,
        eob=0x55555555u,all_bits=0xffffffffu};};

template<>
struct bit_info<npy_uint16>
{	enum {ld_bits=4,bits=16,eob=0x5555,all_bits=0xffff};};

template<>
struct bit_info<npy_uint8>
{	enum {ld_bits=3,bits=8,eob=0x55,all_bits=0xff};};



npy_uint8 bit_count(npy_uint8 I, int l){
  I &= (0x7f >> (7-l));
  I = I - ((I >> 1) & 0x55);
  I = (I & 0x33) + ((I >> 2) & 0x33);
  return (((I + (I >> 4)) & 0x0f) * 0x01); 
}

npy_uint16 bit_count(npy_uint16 I, int l){
  I &= (0x7fff >> (15-l));
  I = I - ((I >> 1) & 0x5555);
  I = (I & 0x3333) + ((I >> 2) & 0x3333);
  return (((I + (I >> 4)) & 0x0f0f) * 0x0101) >> 8;
}

npy_uint32 bit_count(npy_uint32 I, int l){
  I &= (0x7fffffff >> (31-l));
  I = I - ((I >> 1) & 0x55555555);
  I = (I & 0x33333333) + ((I >> 2) & 0x33333333);
  return (((I + (I >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;    
}

npy_uint64 bit_count(npy_uint64 I, int l){
  I &= (0x7fffffffffffffff >> (63-l));
  I = I - ((I >> 1) & 0x5555555555555555);
  I = (I & 0x3333333333333333) + ((I >> 2) & 0x3333333333333333);
  return (((I + (I >> 4)) & 0x0f0f0f0f0f0f0f0f) * 0x0101010101010101) >> 56;
}





#endif
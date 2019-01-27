#ifndef __bitcount_h__
#define __bitcount_h__

#include <stdint.h>


int bitcount_32_C(uint32_t i,int l) {
	i = i & ((0x7FFFFFFF) >> (31 - l));
	#if defined(__GNUC__) || defined(__GNUG__)
		return __builtin_popcount(i);
	#else
		i = i - ((i >> 1) & 0x55555555);
		i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
		return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	#endif
}

int bitcount_64_C(uint64_t i, int l) {
	i = i & ((0x7FFFFFFFFFFFFFFF) >> (63 - l));
	#if defined(__GNUC__) || defined(__GNUG__)
		return __builtin_popcountll(i);
	#else
		i = i - ((i >> 1) & 0x5555555555555555);
		i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
		return (int) ((((i + (i >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56);
	#endif
}

#endif
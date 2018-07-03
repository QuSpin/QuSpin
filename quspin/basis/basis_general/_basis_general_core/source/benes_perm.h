#ifndef _BENES_PERM_H
#define _BENES_PERM_H


#include "numpy/ndarraytypes.h"

#define no_index int(-1)


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



/*

template<typename I>
struct tr_bfly{
  // This structure is used to hold the configuration of
  // butterfly-based operations as well as compress and expand.
  enum {ld_bits = bit_info<I>::ld_bits};
  I cfg[ld_bits];  // butterfly configuration
  I mask;  // saved mask, for compress/expand

  // Here is sketched how to convert this to a class:
  // Include all the generator and usage functions as private methods
  // and replace the parameter self by the implicit object pointer this.
  // Add the many compound routines.
  // Remove the name suffix  for all methods.
  // If you want to cache the configuration, add here:
  //   kind: the generator kind
  //     enum (initialized, frot, vrot, ce_right, ce_left, cef_right, cef_left)
  //   sw: the used subword size (t_subword)
  // Add an initializer/constructor which sets kind to initialized.
  // The generator routines must set the keys (kind, mask, sw).
  // The compound routines check the cached keys (kind, mask, sw);
  //   if not equal, call the generator routine and update the configuration;
  // finally they call the usage routine.
  };

template<typename I>
struct tr_benes{
  tr_bfly<I> b1,b2;
  };

template<typename I>   // primary template
struct ta_index
{
  enum { bits = bit_info<I>::bits};
  int data[bits];

  int& operator[](int idx) {
    return data[idx];
    }

  int operator[](int idx) const {
        return data[idx];
    }
};

template<typename I>   // primary template
struct ta_subword
{
  enum { ld_bits = bit_info<I>::ld_bits};
  unsigned int data[ld_bits];

  unsigned int& operator[](int idx) {
    return data[idx];
    }

  unsigned int operator[](int idx) const {
        return data[idx];
    }
};

//////
// aux functions


template<typename I>
void invert_perm(const ta_index<I> &src, ta_index<I> &tgt) {

  int i;

  for (i = 0; i <= bit_info<I>::bits-1; ++i) {
    tgt[i] = no_index;
    }
  for (i = 0; i <= bit_info<I>::bits-1; ++i) {
    if (src[i] != no_index) {
      tgt[src[i]] = i;
      }
    }
  }

template<typename I>
I bit_permute_step(I x, I m, I shift) {
// INLINE
// Can be replaced by bit_permute_step_simple,
// if for the relevant bits n the following holds:
// nr_1bits(bit_permute_step_simple(n,m,shift)) == nr_1bits(n)
// x86: >= 6/5 cycles
// ARM: >= 4/4 cycles

  // assert(((m << shift) & m) == 0);
  // assert(((m << shift) >> shift) == m);
  I t = ((x >> shift) ^ x) & m;
  x = x ^ t;  t = t << shift;  x = x ^ t;  // x = (x ^ t) ^ (t << shift);
  return x;
  }


//////
// Butterfly network
template<typename I>
I bfly(const tr_bfly<I>* self, I x) {
// Apply butterfly network on x configured by
//   - gen_frot
//   - gen_vrot
//   - gen_cef_right
//   - gen_cef_left

  int stage,j;

  for (stage = bit_info<I>::ld_bits-1; stage >= 0; --stage) {  // UNROLL
    // x = butterfly(x, self->cfg[stage], sw);
    j = 1 << stage;
    x = bit_permute_step<I>(x, self->cfg[stage], j);
    }

  return x;
  }

template<typename I>
I ibfly(const tr_bfly<I>* self, I x) {
// Apply inverse butterfly network on x configured by
//   - gen_frot
//   - gen_vrot
//   - gen_cef_right
//   - gen_cef_left

  int stage,j;

  for (stage = 0; stage <= bit_info<I>::ld_bits-1; ++stage) {  // UNROLL
    // x = butterfly(x, self->cfg[stage], sw);
    j = 1 << stage;
    x = bit_permute_step<I>(x, self->cfg[stage], j);
    }

  return x;
  }


//////
// Permutations via Benes network

template<typename t_bit_index>
static void exchange_bit_index(t_bit_index* a, t_bit_index* b) {
// INLINE

  t_bit_index q;

  q = *a;
  *a = *b;
  *b = q;
  }

template<typename I>
void gen_benes_ex(tr_benes<I>* self, const ta_index<I> c_tgt, const ta_subword<I> a_stage) {
// Generate a configuration for the Benes network with variable stage order.
// Use benes_fwd_ex and benes_bwd_ex.
// Algorithm as sketched by Donal E. Knuth,
//   The art of computer programming, vol. 4, pre-fascicle 1a
// Implemented 2011-12-02 by Jasper L. Neumann
// Modified 2012-08-31 to allow for "don't care" entries.
// Modified 2018-05-22 to allow for templated types only for benes'/butterfly networks.

  ta_index<I> src, inv_src;
  ta_index<I> tgt, inv_tgt;
  int stage;
  int mask;
  I cfg_src,cfg_tgt;
  I src_set;
  int main_idx, aux_idx, src_idx, tgt_idx, idx2;
  int stage_idx;
  int s;
  I lo_bit = 1;

  for (s = 0; s <= bit_info<I>::bits-1; ++s) {
    src[s] = no_index;
    tgt[s] = no_index;
    }
  for (s = 0; s <= bit_info<I>::bits-1; ++s) {
    if (c_tgt[s] != no_index) {
      tgt[s] = s;
      src[c_tgt[s]] = s;
      }
    }
  invert_perm(src,inv_src);
  invert_perm(tgt,inv_tgt);
  for (stage_idx = 0; stage_idx <= bit_info<I>::ld_bits-1; ++stage_idx) {
    stage = a_stage[stage_idx];
    src_set = 0;
    src_idx = 0;
    mask = lo_bit << stage;
    cfg_src = 0;
    cfg_tgt = 0;
    for (main_idx = 0; main_idx <= bit_info<I>::bits-1; ++main_idx) {  // This order to meet Waksman test
      if ((main_idx & mask) == 0) {  // low only
        for (aux_idx = 0; aux_idx <= 1; ++aux_idx) {
          src_idx = main_idx+(aux_idx << stage);
          if (((lo_bit << src_idx) & src_set) == 0) {  // yet unhandled
            if (src[src_idx] != no_index) {  // not open

              do {
                src_set = (src_set | (lo_bit << src_idx));
                tgt_idx = inv_tgt[src[src_idx]];
                if (tgt[tgt_idx] == no_index) {
                  break;  // open end
                  }

                if (((src_idx ^ tgt_idx) & mask) == 0) {
                  // straight
                  tgt_idx = tgt_idx ^ mask;
                  }
                else {
                  // cross
                  cfg_tgt = (cfg_tgt | (lo_bit << (tgt_idx & ~mask)));
                  idx2 = (tgt_idx ^ mask);
                  exchange_bit_index(&tgt[tgt_idx], &tgt[idx2]);
                  inv_tgt[tgt[idx2]] = idx2;
                  if (tgt[tgt_idx] != no_index) {
                    inv_tgt[tgt[tgt_idx]] = tgt_idx;
                    }
                  }

                if (tgt[tgt_idx] == no_index) {
                  break;  // open end
                  }

                src_idx = inv_src[tgt[tgt_idx]];

                if (((src_idx ^ tgt_idx) & mask) == 0) {
                  // straight
                  src_set = (src_set | (lo_bit << src_idx));
                  src_idx = (src_idx ^ mask);
                  }
                else {
                  // cross
                  
                  cfg_src = (cfg_src | (lo_bit << (src_idx & ~mask)));
                  idx2 = (src_idx ^ mask);
                  src_set = (src_set | (lo_bit << idx2));

                  exchange_bit_index(&src[src_idx], &src[idx2]);
                  inv_src[src[idx2]] = idx2;
                  if (src[src_idx] != no_index) {
                    inv_src[src[src_idx]] = src_idx;
                    }
                  }
                if (src[src_idx] == no_index) {
                  break;  // open end
                  }
                if (((lo_bit << src_idx) & src_set) != 0) {  // yet unhandled
                  break;  // already handled
                  }
                } while (true);

              }
            }
          }
        }
      }
    self->b1.cfg[stage] = cfg_src;
    self->b2.cfg[stage] = cfg_tgt;
    }
  // Reduce inner stages to one (not needed)
  // self->b2.cfg[0] = self->b2.cfg[0] ^ self->b1.cfg[0];
  // self->b1.cfg[0] = 0;
  }

template<typename I>
void gen_benes(tr_benes<I>* self, const ta_index<I> c_tgt) {
// INLINE
// Generate a configuration for the standard Benes network.
  ta_subword<I> a_stage_bwd;
  for(unsigned int i=0;i < bit_info<I>::ld_bits;++i){
  	a_stage_bwd[i] = bit_info<I>::ld_bits - i - 1;
  }
  gen_benes_ex(self,c_tgt,a_stage_bwd);  // standard Benes order
  }

template<typename I>
I benes_fwd(const tr_benes<I>* self, I x) {
// Apply Benes network.
// c_tgt of gen_benes selected source indexes.

  return ibfly(&self->b2, bfly(&self->b1,x));
  }

template<typename I>
I benes_bwd(const tr_benes<I>* self, I x) {
// Apply Benes network.
// c_tgt of gen_benes selected target indexes.

  return ibfly(&self->b1, bfly(&self->b2,x));
  }
*/
#endif
#ifndef _SPINLESS_FERMION_BASIS_OP_H
#define _SPINLESS_FERMION_BASIS_OP_H

#include <complex>
#include "hcb_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "openmp.h"
#include "misc.h"

namespace basis_general {

/*
template<class I>
void mergeSort(I nums[],I work[],const I left,const I mid,const I right, bool  &f_count){
    I leftLength = mid - left + 1;
    I rightLength = right - mid;
    I * lAr = work;
    I * rAr = work+leftLength;

    for (I i = 0; i < leftLength; i++) {
      lAr[i] = nums[left + i];
    }
    for (I i = 0; i < rightLength; i++) {
      rAr[i] = nums[mid + 1 + i];
    }
    I i = 0, j = 0, k = left;
    while (i < leftLength && j < rightLength) {
      if (lAr[i] >= rAr[j]) {
        nums[k] = lAr[i];
        if(j&1){f_count ^= 1;}
        i++;
      } else {
        nums[k] = rAr[j];
        j++;
      }
      k++;
    }
    //remaining isertions
    if((j&1) && ((leftLength-i)&1)){f_count ^= 1;}
    if (i >= leftLength) {
      //copy remaining elements from right
      for (; j < rightLength; j++, k++) {
        nums[k] = rAr[j];
      }
    } else {
      //copy remaining elements from left
      for (; i < leftLength; i++, k++) {
        nums[k] = lAr[i];
      }
    }
  }

 //I sort the array using merge sort technique.
template<class I>
void getf_count(I nums[],I work[], I left, I right, bool &f_count){
    if (left < right) {
      I mid = (I)((int)left + (int)right) / 2;
      getf_count(nums, work, left, mid, f_count);
      getf_count(nums, work, (I)(mid + 1), right, f_count);
      mergeSort(nums, work, left, mid, right, f_count);
    }
  }
*/




template<class I,class P>
I inline spinless_fermion_map_bits(I s,const int map[],const int N,P &sign){
	I ss = 0;
	int np = 0;
	int pos_list[bit_info<I>::bits];
	

	for(int i=N-1;i>=0;--i){
		const int j = map[i];
		const I n = (s&1);
		const bool neg = j<0;

		if(n){
			pos_list[np++] = ( neg ? -(j+1) : j);
			// f_count ^= (neg&&(i&1)); do not change sign based on PH transformation
		}
		ss ^= ( neg ? (n^1)<<(N+j) : n<<(N-j-1) );

		s >>= 1;
	}

	// getf_count(pos_list,work,0,np-1,f_count);
	int Nswap = countSwaps<I>(pos_list,np);
	if(Nswap&1){sign *= -1;}

	return ss;
}

template<class I,class P>
void get_map_sign(I s,I inv,P &sign){
	typename bit_info<I>::bit_index_type pos_list[bit_info<I>::bits];
	bool f_count = 0;

	I ne = bit_count(bit_info<I>::eob&s,bit_info<I>::bits-1); // count number of partices on odd sites
	f_count ^= (ne&1); 
	typename bit_info<I>::bit_index_type n = bit_pos(s,pos_list) - 1; // get bit positions
	getf_count(pos_list,(typename bit_info<I>::bit_index_type)0,n,f_count);
	if(f_count){sign *= -1;}	
}


template<class I,class P=signed char>
class spinless_fermion_basis_core : public hcb_basis_core<I,P>
{

	public:
		spinless_fermion_basis_core(const int _N,const bool _pre_check=false) : \
		hcb_basis_core<I>::hcb_basis_core(_N,true,_pre_check) { }

		spinless_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
						   const int _pers[], const int _qs[],const bool _pre_check=false) : \
		hcb_basis_core<I>::hcb_basis_core(_N,_nt,_maps,_pers,_qs,true,_pre_check) {}

		~spinless_fermion_basis_core(){}


		// I map_state(I s,int n_map,int &sign){
		// 	if(general_basis_core<I,P>::nt<=0){
		// 		return s;
		// 	}
		// 	get_map_sign<I>(s,hcb_basis_core<I>::invs[n_map],sign);
		// 	return benes_bwd(&hcb_basis_core<I>::benes_maps[n_map],s^hcb_basis_core<I>::invs[n_map]);;
			
		// }

		// void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
		// 	if(general_basis_core<I,P>::nt<=0){
		// 		return;
		// 	}
		// 	const tr_benes<I> * benes_map = &hcb_basis_core<I>::benes_maps[n_map];
		// 	const I inv = hcb_basis_core<I>::invs[n_map];
		// 	#pragma omp for schedule(static,1)
		// 	for(npy_intp i=0;i<M;i++){
		// 		int temp_sign = sign[i];
		// 		get_map_sign<I>(s[i],inv,temp_sign);
		// 		s[i] = benes_bwd(benes_map,s[i]^inv);
		// 		sign[i] = temp_sign;
		// 	}
		// }

		I map_state(I s,int n_map,P &sign){
			if(general_basis_core<I,P>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I,P>::N;
			return spinless_fermion_map_bits(s,&general_basis_core<I,P>::maps[n_map*n],n,sign);
			
		}

		void map_state(I s[],npy_intp M,int n_map,P sign[]){
			if(general_basis_core<I,P>::nt<=0){
				return;
			}
			const int n = general_basis_core<I,P>::N;
			const int * map = &general_basis_core<I,P>::maps[n_map*n];
			#pragma omp for schedule(static)
			for(npy_intp i=0;i<M;i++){
				s[i] = spinless_fermion_map_bits(s[i],map,n,sign[i]);
			}
		}


		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			const I s = r;
			const I one = 1;
			for(int j=n_op-1;j>-1;j--){
				
				const int ind = general_basis_core<I,P>::N-indx[j]-1;
				I f_count = bit_count(r,ind);
				double sign = ((f_count&1)?-1:1);
				const I b = (one << ind);
				const bool a = (bool)((r >> ind)&one);
				const char op = opstr[j];
				switch(op){
					case 'z':
						m *= (a?0.5:-0.5);
						break;
					case 'x':
						m *= sign;
						r ^= b;
						break;
					case 'y': // corresponds to -\sigma^y
						m *= (a?std::complex<double>(0,-1.0*sign):std::complex<double>(0,+1.0*sign));
						r ^= b;
						break;
					case 'n':
						m *= (a?1:0);
						break;
					case '+':
						m *= (a?0:sign);
						r ^= b;
						break;
					case '-':
						m *= (a?sign:0);
						r ^= b;
						break;
					case 'I':
						break;
					default:
						return -1;
				}

				if(m.real()==0 && m.imag()==0){
					r = s;
					break;
				}
			}

			return 0;
		}
};



}



#endif

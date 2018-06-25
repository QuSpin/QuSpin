#ifndef _SPINLESS_FERMION_BASIS_OP_H
#define _SPINLESS_FERMION_BASIS_OP_H

#include <complex>
#include "hcb_basis_core.h"
#include "numpy/ndarraytypes.h"




void mergeSort(int nums[], int left, int mid, int right, bool  &f_count){
    int leftLength = mid - left + 1;
    int rightLength = right - mid;
    int lAr[64];
    int rAr[64];
    for (int i = 0; i < leftLength; i++) {
      lAr[i] = nums[left + i];
    }
    for (int i = 0; i < rightLength; i++) {
      rAr[i] = nums[mid + 1 + i];
    }
    int i = 0, j = 0, k = left;
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
    //remaining iversions
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
void getf_count(int nums[], int left, int right, bool &f_count){
    if (left < right) {
      int mid = (left + right) / 2;
      getf_count(nums, left, mid, f_count);
      getf_count(nums, mid + 1, right, f_count);
      mergeSort(nums, left, mid, right, f_count);
    }
  }




// template<class I>
// I inline spinless_fermion_map_bits(I s,const int map[],const int N,int &sign){
// 	I ss = 0;
// 	int pos_list[64];
// 	int np = 0;
// 	I sf_count = ((inv&s) & (bit_info<I>::eob >> (N&1)));
// 	bool f_count = (bit_count(sf_count,0)&1);

// 	s ^= inv;

// 	for(int i=N-1;i>=0;--i){
// 		if(s&1){
// 			pos_list[np]=j;++np;
// 			ss ^= ( I(1)<<(N-map[i]-1) );
// 		}
// 		s >>= 1;
// 	}

// 	if(np>1){getf_count(pos_list,0,np-1,f_count);}
// 	if(f_count){sign ^= (-2);}

// 	return ss;
// }



template<class I>
I inline spinless_fermion_map_bits(I s,const int map[],const int N,int &sign){
	I ss = 0;
	int pos_list[64];
	int np = 0;
	bool f_count = 0;

	for(int i=N-1;i>=0;--i){
		int j = map[i];
		I n = (s&1);
		if(n){
			pos_list[np] = ( j<0 ? -(j+1) : j);
			++np;
			f_count ^= ((j<0)&&(i&1));
		}
		ss ^= ( j<0 ? (n^1)<<(N+j) : n<<(N-j-1) );

		s >>= 1;
	}

	getf_count(pos_list,0,np-1,f_count);
	if(f_count){sign *= -1;}

	return ss;
}


template<class I>
class spinless_fermion_basis_core : public hcb_basis_core<I>
{
	public:
		spinless_fermion_basis_core(const int _N) : \
		hcb_basis_core<I>::hcb_basis_core(_N) {}

		spinless_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
						   const int _pers[], const int _qs[]) : \
		hcb_basis_core<I>::hcb_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~spinless_fermion_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N;
			return spinless_fermion_map_bits(s,&general_basis_core<I>::maps[n_map*n],n,sign);
			
		}

		void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			const int n = general_basis_core<I>::N;
			const int * map = &general_basis_core<I>::maps[n_map*n];
			#pragma omp for schedule(static,1)
			for(npy_intp i=0;i<M;i++){
				int temp_sign = sign[i];
				s[i] = spinless_fermion_map_bits(s[i],map,n,temp_sign);
				sign[i] = temp_sign;
			}
		}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			I one = 1;

			for(int j=n_op-1;j>-1;j--){
				int ind = general_basis_core<I>::N-indx[j]-1;
				I f_count = bit_count(r,ind);
				double sign = ((f_count&1)?-1:1);
				I b = (one << ind);
				bool a = bool((r >> ind)&one);
				char op = opstr[j];
				switch(op){
					case 'z':
						m *= (a?0.5:-0.5);
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

				if(std::abs(m)==0){
					r = s;
					break;
				}
			}

			return 0;
		}
};







#endif

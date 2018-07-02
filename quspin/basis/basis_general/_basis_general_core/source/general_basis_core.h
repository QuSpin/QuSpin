#ifndef _GENERAL_BASIS_CORE_H
#define _GENERAL_BASIS_CORE_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <complex>
#include <vector>
#include <stdlib.h>
#include "numpy/ndarraytypes.h"
#include "benes_perm.h"


template<class I>
class general_basis_core{
	public:
		const int N;
		const int nt;
		const int * maps;
		const int * pers;
		const int * qs;

		general_basis_core(const int _N) : \
			 N(_N), nt(0), maps(NULL), pers(NULL), qs(NULL) {}

		general_basis_core(const int _N,const int _nt,const int _maps[], \
			const int _pers[], const int _qs[]) : \
			 N(_N), nt(_nt), maps(_maps) , pers(_pers), qs(_qs) { }

		~general_basis_core() {}

		double check_state(I);
		I ref_state(I,int[],int[],int&);
		virtual I next_state_pcon(I) = 0;
		virtual int op(I&,std::complex<double>&,const int,const char[],const int[]) = 0;
		virtual void map_state(I[],npy_intp,int,signed char[]) = 0;
		virtual I map_state(I,int,int&) = 0;
		// virtual void print(I) = 0;
		virtual int get_N() const{
			return N;
		}

		virtual int get_nt() const{
			return nt;
		}
};




template<class I>
double check_state_core(general_basis_core<I> *B,I t,int sign, double k, double norm,const I s,const int nt,const int depth){
	if(nt<=0){
		return 1;
	}

	const int per = B->pers[depth];
	const double q = (2.0*M_PI*B->qs[depth])/per;

	if(depth<nt-1){
		for(int i=1;i<per+1;i++){
			norm = check_state_core(B,t,sign,k,norm,s,nt,depth+1);
			k += q;
			t = B->map_state(t,depth,sign);

			#if defined(_WIN64)
				// x64 version
				bool isnan = _isnanf(norm) != 0;
			#elif defined(_WIN32)
				bool isnan = _isnan(norm) != 0;
			#else
				bool isnan = std::isnan(norm);
			#endif

			if(t > s || isnan){
				return std::numeric_limits<double>::quiet_NaN();
			}
		}
	}
	else{
		for(int i=1;i<per+1;i++){

			if(t==s){
				norm += sign * std::cos(k);
			}
			k += q;
			t = B->map_state(t,depth,sign);
			// std::cout << s << "," << t << "," << sign << ",   ";
			if(t > s){
				return std::numeric_limits<double>::quiet_NaN();
			}
		}
	}

	return norm;
}

template<class I>
I ref_state_core(general_basis_core<I> *B, const I s,I r,int g[], int gg[],int &sign,int &tempsign,const int nt,const int depth){
	if(nt<=0){
		return s;
	}
	I t = s;
	const int per = B->pers[depth];
	if(depth<nt-1){
		for(int i=0;i<per;i++){
			gg[depth] = i;
			r = ref_state_core(B,t,r,g,gg,sign,tempsign,nt,depth+1);
			t = B->map_state(t,depth,tempsign);
		}
		return r;
	}
	else{
		for(int i=0;i<per;i++){
			gg[depth] = i;
			if(t>r){
				r = t;
				sign = tempsign;
				for(int j=0;j<nt;j++){
					g[j] = gg[j];
				}
			}
			t = B->map_state(t,depth,tempsign);
		}
		return r;
	}
}

template<class I>
double general_basis_core<I>::check_state(I s){
	int sign=1;
	double k=0.0;
	double norm=0.0;
	return check_state_core<I>(this,s,sign,k,norm,s,nt,0);
}

template<class I>
I general_basis_core<I>::ref_state(I s,int g[],int gg[],int &sign){
	int tempsign=1;
	for(int i=0;i<nt;i++){
		g[i] = 0;
		gg[i] = 0;
	}
	return ref_state_core<I>(this,s,s,g,gg,sign,tempsign,nt,0);
}



// template<class I>
// bool check_state_core(general_basis_core<I> *B,I t,int sign,const I s,const int nt,const int depth){
// 	if(nt<=0){
// 		return true;
// 	}
// 	const int per = B->pers[depth];

// 	if(depth<nt-1){
// 		for(int i=1;i<per+1;i++){
// 			if(!check_state_core(B,t,sign,s,nt,depth+1))
// 				return false;

// 			t = B->map_state(t,depth,sign);
// 			if(t > s){
// 				return false;
// 			}
// 			else if (t == s){
// 				break;
// 			}

// 		}
// 		return true;
// 	}
// 	else{
// 		for(int i=1;i<per+1;i++){
// 			t = B->map_state(t,depth,sign);
// 			if(t > s){
// 				return false;
// 			}
// 			else if (t == s){
// 				break;
// 			}
// 		}
// 		return true;
// 	}
// }


// template<class I>
// double get_norm_core(general_basis_core<I> *B,I t,int sign, double k, double norm, const I s,const int nt,const int depth){
// 	if(nt<=0){
// 		return 1;
// 	}
// 	const int per = B->pers[depth];
// 	const double q = (2.0*M_PI*B->qs[depth])/per;

// 	if(depth<nt-1){
// 		for(int i=0;i<per;i++){
// 			norm = get_norm_core(B,t,sign,k,norm,s,nt,depth+1);
// 			k += q;
// 			t = B->map_state(t,depth,sign);
// 		}
// 	}
// 	else{
// 		for(int i=0;i<per;i++){
// 			if(t==s){
// 				norm += sign * std::cos(k);
// 			}
// 			k += q;
// 			t = B->map_state(t,depth,sign);
// 		}
// 	}

// 	return norm;
// }



#endif

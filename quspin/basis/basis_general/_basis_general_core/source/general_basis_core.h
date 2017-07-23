#ifndef _GENERAL_BASIS_CORE_H
#define _GENERAL_BASIS_CORE_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <stdlib.h>
#include "numpy/ndarraytypes.h"



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
			 N(_N), nt(_nt), maps(_maps), pers(_pers), qs(_qs) {}

		~general_basis_core() {}

		virtual bool check_state(I) = 0;
		virtual I ref_state(I,int[],int[]) = 0;
		virtual double get_norm(I) = 0;
		virtual I next_state_pcon(I) = 0;
		virtual int op(I&,std::complex<double>&,const int,const char[],const int[]) = 0;
		virtual void map_state(I[],npy_intp,int) = 0;
		virtual I map_state(I,int) = 0;
		virtual void print(I) = 0;
		virtual int get_N() const{
			return N;
		}

		virtual int get_nt() const{
			return nt;
		}
};



template<class I>
bool check_state_core(general_basis_core<I> *B,I t,const I s,const int nt,const int depth){
	if(nt<=0){
		return true;
	}
	const int per = B->pers[depth];
	const int q = B->qs[depth];
	bool keep = true;
	if(depth<nt-1){
		for(int i=1;i<per+1 && keep ;i++){
			keep = keep && check_state_core(B,t,s,nt,depth+1);
			t = B->map_state(t,depth);
			if(t < s){
				return false;
			}
			else if(t==s){
				if(q%(per/i) && per>2) return false;
				break;
			}
		}
		return keep;
	}
	else{
		for(int i=1;i<per+1;i++){
			t = B->map_state(t,depth);
			if(t < s){
				return false;
			}
			else if(t==s){
				if(q%(per/i) && per>2) return false;
				break;
			}
		}
		return true;
	}
}

template<class I>
double get_norm_core(general_basis_core<I> *B,I s1,I s2,const int nt,const int depth){
	if(nt<=0){
		return 1;
	}
	const int per = B->pers[depth];
	const double q = (2.0*M_PI*B->qs[depth])/per;
	double norm = 0;

	if(depth<nt-1){
		for(int i=0;i<per;i++){
			for(int j=0;j<per;j++){
				norm += std::cos(q*(i-j))*get_norm_core(B,s1,s2,nt,depth+1);
				s2 = B->map_state(s2,depth);
			}
			s1 = B->map_state(s1,depth);
		}
		return norm;
	}
	else{
		for(int i=0;i<per;i++){
			for(int j=0;j<per;j++){
				if(s1==s2){
					norm += std::cos(q*(i-j));
				}
				s2 = B->map_state(s2,depth);
			}
			s1 = B->map_state(s1,depth);
		}
		return norm;
	}
}

template<class I>
I ref_state_core(general_basis_core<I> *B, const I s,I r,int g[], int gg[],const int nt,const int depth){
	if(nt<=0){
		return s;
	}
	I t = s;
	const int per = B->pers[depth];
	if(depth<nt-1){
		for(int i=0;i<per;i++){
			gg[depth] = i;
			r = ref_state_core(B,t,r,g,gg,nt,depth+1);
			t = B->map_state(t,depth);
		}
		return r;
	}
	else{
		for(int i=0;i<per;i++){
			gg[depth] = i;
			if(t<r){
				r = t;
				for(int j=0;j<nt;j++){
					g[j] = gg[j];
				}
			}
			t = B->map_state(t,depth);
		}
		return r;
	}
}


#endif

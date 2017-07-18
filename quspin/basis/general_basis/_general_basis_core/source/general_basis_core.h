#ifndef _GENERAL_BASIS_CORE_H
#define _GENERAL_BASIS_CORE_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <stdlib.h>
#include "numpy/ndarraytypes.h"


template<class I>
using map_func_type = I (*)(I,const int[],const int);

template<class I>
bool check_state_core(const map_func_type<I> map_func,I t,const I s,const int maps[],const int pers[],const int qs[],const int nt,const int N){
	bool keep = true;
	int per = pers[0];
	int q = qs[0];
	if(nt > 1){
		for(int i=1;i<per+1 && keep ;i++){
			keep = keep && check_state_core(map_func,t,s,&maps[N],&pers[1],&qs[1],nt-1,N);
			t = map_func(t,maps,N);
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
	else if(nt==1){
		for(int i=1;i<per+1;i++){
			t = map_func(t,maps,N);
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
	else{
		return true;
	}
}

template<class I>
double get_norm_core(const map_func_type<I> map_func,I s1,I s2,const int maps[], const int rev_maps[],const int pers[],const int qs[],const int nt,const int N){
	double norm = 0;
	const int per = pers[0];
	const double q = (2.0*M_PI*qs[0])/per;

	if(nt > 1){
		for(int i=0;i<per;i++){
			for(int j=0;j<per;j++){
				norm += std::cos(q*(i-j))*get_norm_core(map_func,s1,s2,&maps[N],&rev_maps[N],&pers[1],&qs[1],nt-1,N);
				s2 = map_func(s2,maps,N);
			}
			s1 = map_func(s1,maps,N);
		}
		return norm;
	}
	else if(nt==1){
		for(int i=0;i<per;i++){
			for(int j=0;j<per;j++){
				if(s1==s2){
					norm += std::cos(q*(i-j));
				}
				s2 = map_func(s2,maps,N);
			}
			s1 = map_func(s1,maps,N);
		}
		return norm;

	}
	else{
		return 1.0;
	}
}

template<class I>
I ref_state_core(const map_func_type<I> map_func,const I s,const int maps[],const int pers[],const int nt,const int nnt,const int N,I r,int g[], int gg[]){
	if(nnt > 1){
		I t = s;
		const int per = pers[0];
		for(int i=0;i<per;i++){
			gg[nt-nnt] = i;
			r = ref_state_core(map_func,t,&maps[N],&pers[1],nt,nnt-1,N,r,g,gg);
			t = map_func(t,maps,N);
		}
		return r;
	}
	else if(nnt==1){
		I t = s;
		const int per = pers[0];
		for(int i=0;i<per;i++){
			gg[nt-nnt] = i;

			// std::cout << std::setw(5);
			// for(int ee=0;ee<nt;ee++){std::cout << gg[ee] << std::setw(5);}
			// std::cout << std::endl;

			if(t<r){
				r = t;
				for(int j=0;j<nt;j++){
					g[j] = gg[j];
				}
			}
			t = map_func(t,maps,N);
		}
		return r;
	}
	else{
		return r;
	}
}

template<class I>
class general_basis_core{
	public:
		const int N,nt;
		const int * maps;
		const int * rev_maps;
		const int * pers;
		const int * qs;

		general_basis_core(const int _N,const int _nt,const int _maps[], \
			const int _rev_maps[], const int _pers[], const int _qs[]) : \
			 N(_N), nt(_nt), maps(_maps), rev_maps(_rev_maps), pers(_pers), qs(_qs) {}

		~general_basis_core() {}

		virtual bool check_state(I) = 0;
		virtual I ref_state(I,int[]) = 0;
		virtual double get_norm(I) = 0;
		virtual I next_state_pcon(I) = 0;
		virtual int op(I&,std::complex<double>&,const int,const unsigned char[],const int[]) = 0;
		
		virtual void map_state(I[],npy_intp,int) = 0;

		virtual int get_N() const{
			return N;
		}

		virtual int get_nt() const{
			return nt;
		}
};

#endif

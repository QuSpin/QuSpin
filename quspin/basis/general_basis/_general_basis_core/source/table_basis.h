#ifndef _EXPM_MULTIPLY_H
#define _EXPM_MULTIPLY_H

#include <cmath>
#include <iostream>
#include <omp.h>



size_t comb(const int N,const int k){
	const int M = N + 1;
	const int nterms = (k<(N - k)) ? k : N-k;

	if(k<0 || k>N){return 0;}

	size_t val = 1;
	for(int j=1;j<nterms+1;j++){
		val *= M - j;
		val /= j;
	}

	return val;
}


template<class I>
I inline map_bits(I s,const int map[],const int N){
	I ss = 0;
	for(int i=0;i<N;i++){
		int j = map[i];
		ss ^= ( j<0 ? ((s&1)^1)<<(-(j+1)) : (s&1)<<j );
		s >>= 1;
	}
	return ss;
}


template<class I>
bool check_state_core(I t,const I s,const int maps[],const int pers[],const int qs[],const int nt,const int N){
	bool keep = true;
	int per = pers[0];
	int q = qs[0];
	if(nt > 1){
		for(int i=1;i<per+1 && keep ;i++){
			keep = keep && check_state_core(t,s,&maps[N],&pers[1],&qs[1],nt-1,N);
			if(keep){
				t = map_bits(t,maps,N);
				if(t < s){
					return false;
				}
				else if(t==s){
					if(q%per/i != 0) return false;
					break;
				}
			}
		}
		return keep;
	}
	else{
		for(int i=1;i<per+1;i++){
			t = map_bits(t,maps,N);
			if(t < s){
				return false;
			}
			else if(t==s){
				if(q%per/i != 0) return false;
				break;
			}
		}
		return true;
	}
}

template<class I>
double get_norm_core(I s1,I s2,const int maps[], const int rev_maps[],const int pers[],const int qs[],const int nt,const int N){
	double norm = 0;
	const int per = pers[0];
	const double q = (2.0*M_PI*qs[0])/per;

	if(nt > 1){
		for(int i=0;i<per;i++){
			for(int j=0;j<per;j++){
				if(s1==s2){
					norm += std::cos(q*(i-j))*get_norm_core(s1,s2,&maps[N],&rev_maps[N],&pers[1],&qs[1],nt-1,N);
				}
				s2 = map_bits(s2,maps,N);
			}
			s1 = map_bits(s1,maps,N);
		}
		return norm;
	}
	else{
		for(int i=0;i<per;i++){
			for(int j=0;j<per;j++){
				if(s1==s2){
					norm += std::cos(q*(i-j));
				}
				s2 = map_bits(s2,maps,N);
			}
			s1 = map_bits(s1,maps,N);
		}
		return norm;

	}
}

template<class I>
I ref_state_core(const I s,const int maps[],const int pers[],const int nt,const int nnt,const int N,I r,int g[], int gg[]){
	I t = s;
	const int per = pers[0];

	if(nnt > 1){
		for(int i=0;i<per;i++){
			gg[nt-nnt] = i;
			r = ref_state_helper(t,&maps[N],&pers[1],nt,nt-1,N,r,g,gg);
			t = map_bits(t,maps,N);
		}
		return r;
	}
	else{
		for(int i=0;i<per;i++){
			gg[nt-nnt] = i;
			if(t<r){
				r = t;
				for(int j=0;j<nt;j++){
					g[j] = gg[j];
				}
			}
			t = map_bits(t,maps,N);
		}
		return r;
	}
}


template<class I>
class general_basis_core{
	private:
		const int N,nt;
		const int * maps;
		const int * rev_maps;
		const int * pers;
		const int * qs;


	public:
		general_basis_core(const int _N,const int _nt,const int _maps[], \
			const int _rev_maps[], const int _pers[], const int _qs[]) : \
			 N(_N), nt(_nt), maps(_maps), rev_maps(_rev_maps), pers(_pers), qs(_qs) {}

		~general_basis_core() {}

		bool check_state(I s){
			return check_state_core(s,s,maps,pers,qs,nt,N);
		}
		I ref_state(I s,int g[]){
			int gg[nt];
			return ref_state_core(s,maps,pers,nt,nt,N,s,g,gg);
		}
		double get_norm(I s){
			return get_norm_core(s,s,maps,rev_maps,pers,qs,nt,N);
		}

		int get_N(){
			return N;
		}
};

template<class I,class J>
size_t hcb_get_basis(general_basis_core<I> &B,I basis[],J n[]){
	size_t MAX = (1<<B.get_N());
	size_t ii = 0;

	for(size_t s=0;s<MAX;s++){
		if(B.check_state(s)){
			J nn = B.get_norm(s);
			if(n>0){
				basis[ii] = s;
				n[ii] = nn;
				ii++;
			}
		}
	}
	return ii;
}

template<class I,class J>
size_t hcb_pcon_get_basis(general_basis_core<I> &B,int Nup,I basis[],J n[]){
	size_t MAX = comb(B.get_N(),Nup);
	size_t ii = 0;
	I s = 0;
	for(int i=0;i<Nup;i++)
		s += (1ull<<i);

	for(size_t i=0;i<MAX;i++){
		if(B.check_state(s)){
			J nn = B.get_norm(s);
			if(n>0){
				basis[ii] = s;
				n[ii] = nn;
				ii++;
			}
		}
		I t = (s | (s - 1)) + 1;
		s = t | ((((t & -t) / (s & -s)) >> 1) - 1);
	}

	return ii;
}



#endif

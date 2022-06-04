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
#include "bits_info.h"
#include <set>

#define __GENERAL_BASIS_CORE__max_nt 32


namespace basis_general {

template<class I,class P=signed char>
class general_basis_core{
	public:
		const int N;
		const int nt;
		const int * maps;
		const int * pers;
		const int * qs;
		const bool fermionic,pre_check;

		general_basis_core(const int _N, const bool _fermionnic=false, const bool _pre_check=false) : \
			 N(_N), nt(0), maps(NULL), pers(NULL), qs(NULL), fermionic(_fermionnic), pre_check(_pre_check) {}

		general_basis_core(const int _N,const int _nt,const int _maps[], \
			const int _pers[], const int _qs[], const bool _fermionnic=false, const bool _pre_check=false) : \
			 N(_N), nt(_nt), maps(_maps) , pers(_pers), qs(_qs), fermionic(_fermionnic), pre_check(_pre_check) { }

		~general_basis_core() {}

		bool check_pcon(const I,const std::set<std::vector<int>>&);
		virtual double check_state(I);
		I ref_state(I,int[],P&);
		virtual I next_state_pcon(I,I) = 0;
		virtual int op(I&,std::complex<double>&,const int,const char[],const int[]) = 0;
		virtual void map_state(I[],npy_intp,int,P[]) = 0;
		virtual I map_state(I,int,P&) = 0;
		virtual std::vector<int> count_particles(const I s) = 0;
		virtual npy_intp get_prefix(const I s,const int N_p) = 0;
		// virtual void print(I) = 0;
		virtual int get_N() const{
			return N;
		}

		virtual int get_nt() const{
			return nt;
		}
};

template<class T>
inline double get_real(T val){
	return (double)val;
}

template<>
inline double get_real(std::complex<double> val){
	return std::real(val);
}



template<class I,class P=signed char>
double check_state_core_unrolled(general_basis_core<I,P> *B,const I s,const int nt){

	if(nt <= 0 || nt > __GENERAL_BASIS_CORE__max_nt){
		return 1;
	}

	int gg[__GENERAL_BASIS_CORE__max_nt];
	double ks[__GENERAL_BASIS_CORE__max_nt];
	double k = 0;
	P phase = 1;
	double norm = 0;
	I t = s;


	bool status = false;
	int MAXVALSSUM = 0;

	for (int depth=0;depth<nt;depth++){
		gg[depth] = 0;  // Initialize values
		ks[depth] = (2.0 * M_PI * B->qs[depth]) / B->pers[depth];
		MAXVALSSUM += (B->pers[depth]-1);
	}


	while (!status) { 
	
		int total = 0;
		// calculate total for exit condition
		for (int depth=0;depth<nt;depth++){total += gg[depth];}
		// test for exit condition
		if (total >= MAXVALSSUM){status = true;}


		// increment loop variables and transform state
		bool change = true;
		int depth = nt-1;  // start from innermost loop
		while (change && depth>=0) {
			// increment the innermost variable and check if spill overs
			if (++gg[depth] > B->pers[depth]-1) {		
				gg[depth] = 0;  // reintialize loop variable
				// Change the upper variable by one
				// We need to increment the immediate upper level loop by one
				change = true;
			}
			else
				change = false; // Stop as there the upper levels of the loop are unaffected

			// We can perform any inner loop calculation here gg[depth]
			if(depth == (nt-1) && t==s){norm += get_real(phase) * std::cos(k);}
			k += ks[depth];
			t = B->map_state(t,depth,phase);
			if(t > s){
				return std::numeric_limits<double>::quiet_NaN();
			}

			depth--;  // move to upper level of the loop
		}
	}

	return norm;
}


template<class I,class P=signed char>
I ref_state_core_unrolled(general_basis_core<I,P> *B, const I s,int g[],P &phase,const int nt){

	if(nt <= 0 || nt > __GENERAL_BASIS_CORE__max_nt){
		for(int i=0;i<std::min((int)__GENERAL_BASIS_CORE__max_nt,nt);i++){g[i]=0;}
		return s;
	}

	int gg[__GENERAL_BASIS_CORE__max_nt];  // represent the different variables in the for loops;
	P temp_phase = 1;
	I t = s;
	I r = s;


	bool status = false;
	int MAXVALSSUM = 0;

	for (int depth=0;depth<nt;depth++){
		gg[depth] = 0;  // Initialize values
		MAXVALSSUM += (B->pers[depth]-1);
	}


	while (!status) { 
	
		int total = 0;
		// calculate total for exit condition
		for (int depth=0;depth<nt;depth++){total += gg[depth];}
		// test for exit condition
		if (total >= MAXVALSSUM){status = true;}


		// increment loop variables and transform state
		bool change = true;
		int depth = nt-1;  // start from innermost loop
		while (change && depth>=0) {
			if(t>r){
				r = t;
				phase = temp_phase;
				for(int j=0;j<nt;j++){
					g[j] = gg[j];
				}
			}

			// increment the innermost variable and check if spill overs
			if (++gg[depth] > B->pers[depth]-1) {		
				gg[depth] = 0;  // reintialize loop variable
				// Change the upper variable by one
				// We need to increment the immediate upper level loop by one
				change = true;
			}
			else
				change = false; // Stop as there the upper levels of the loop are unaffected

			// We can perform any inner loop calculation here gg[depth]
			t = B->map_state(t,depth,temp_phase);


			depth--;  // move to upper level of the loop
		}
	}

	return r;
}


template<class I,class P>
I general_basis_core<I,P>::ref_state(I s,int g[],P &phase){
	return ref_state_core_unrolled<I,P>(this,s,g,phase,nt);
}



template<class I,class P>
double general_basis_core<I,P>::check_state(I s){
	return check_state_core_unrolled<I,P>(this,s,nt);
}


template<class I,class P>
bool general_basis_core<I,P>::check_pcon(const I s,const std::set<std::vector<int>> &Np){
	// basis_core objects have a count_particles function which returns a vector of the required size;
	// cython construct a vector of vectors, each sub-vector can be arbitrary size: see function load_pcon_list in general_basis_core.pyx
	// in order to be compatible with later general basis classes which may have more than two spcies of particles!
	//
	bool pcon = false;
	std::vector<int> v = this->count_particles(s); 
	typename std::set<std::vector<int>>::iterator it;
	for(it=Np.begin();it!=Np.end();++it){
		pcon |= std::equal(v.begin(),v.end(),(*it).begin());
		
	}
	return pcon;
}



/* ----------------------------------------------------------------------------------------------------------------------------- */


// template<class T>
// bool inline isnan(T val){
// #if defined(_WIN64)
// 	// x64 version
// 	return _isnanf(val) != 0;
// #elif defined(_WIN32)
// 	return _isnan(val) != 0;
// #else
// 	return std::isnan(val);
// #endif
// }

// template<class I>
// double check_state_core(general_basis_core<I> *B,I t,int sign, double k, double norm,const I s,const int nt,const int depth){
// 	if(nt<=0){
// 		return 1;
// 	}

// 	const int per = B->pers[depth];
// 	const double q = (2.0*M_PI*B->qs[depth])/per;

// 	if(depth<nt-1){
// 		for(int i=1;i<per+1;i++){
// 			norm = check_state_core(B,t,sign,k,norm,s,nt,depth+1);
// 			k += q;
// 			t = B->map_state(t,depth,sign);
// 			if(t > s || isnan(norm)){
// 				return std::numeric_limits<double>::quiet_NaN();
// 			}
// 		}
// 	}
// 	else{
// 		for(int i=1;i<per+1;i++){

// 			if(t==s){
// 				norm += sign * std::cos(k);
// 			}
// 			k += q;
// 			t = B->map_state(t,depth,sign);
// 			if(t > s){
// 				return std::numeric_limits<double>::quiet_NaN();
// 			}
// 		}
// 	}

// 	return norm;
// }

// template<class I>
// I ref_state_core(general_basis_core<I> *B, const I s,I r,int g[], int gg[],int &sign,int &temp_phase,const int nt,const int depth){
// 	if(nt<=0){
// 		return s;
// 	}
// 	I t = s;
// 	const int per = B->pers[depth];
// 	if(depth<nt-1){
// 		for(int i=0;i<per;i++){
// 			gg[depth] = i;
// 			r = ref_state_core(B,t,r,g,gg,sign,temp_phase,nt,depth+1);
// 			t = B->map_state(t,depth,temp_phase);
// 		}
// 		return r;
// 	}
// 	else{
// 		for(int i=0;i<per;i++){
// 			gg[depth] = i;
// 			if(t>r){
// 				r = t;
// 				sign = temp_phase;
// 				for(int j=0;j<nt;j++){
// 					g[j] = gg[j];
// 				}
// 			}
// 			t = B->map_state(t,depth,temp_phase);
// 		}
// 		return r;
// 	}
// }




/* ----------------------------------------------------------------------------------------------------------------------------- */

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



// template<class I>
// double general_basis_core<I>::check_state(I s){
// 	int sign=1;
// 	double k=0.0;
// 	double norm=0.0;
// 	return check_state_core<I>(this,s,sign,k,norm,s,nt,0);
// }



// template<class I>
// I general_basis_core<I>::ref_state(I s,int g[],int gg[],int &sign){
// 	int temp_phase=1;
// 	for(int i=0;i<nt;i++){
// 		g[i] = 0;
// 		gg[i] = 0;
// 	}
// 	return ref_state_core<I>(this,s,s,g,gg,sign,temp_phase,nt,0);
// }
}

#endif

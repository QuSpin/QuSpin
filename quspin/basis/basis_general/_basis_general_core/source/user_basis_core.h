#ifndef _user_basis_core_H
#define _user_basis_core_H

#include <complex>
#include <vector>
#include <stdio.h>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "benes_perm.h"
#include "openmp.h"

namespace basis_general {

template<class I>
struct op_results
{
	std::complex<double> m;
	I r;
	op_results(std::complex<double> _m,I _r): m(_m),r(_r)
	{}
};


template<class I>
class user_basis_core : public general_basis_core<I>
{
	typedef I (*map_type)(I,int,int*);
	typedef I (*next_state_type)(I,I,I*);
	typedef int (*op_func_type)(op_results<I>*,char,int,int);
	typedef void (*count_particles_type)(I,int*);
	typedef bool (*check_state_nosymm_type)(I,I,I*);

	public:
		map_type * map_funcs;
		next_state_type next_state_func;
		op_func_type op_func;
		count_particles_type count_particles_func;
		check_state_nosymm_type check_state_nosymm;
		const int n_sectors;
		I *ns_args,*csns_args;


		user_basis_core(const int _N,const int _nt,void * _map_funcs, 
			const int _pers[], const int _qs[], const int _n_sectors,
			size_t _next_state,I *_ns_args,size_t _check_state_nosymm,
			I* _csns_args,size_t _count_particles,size_t _op_func) : \
		general_basis_core<I>::general_basis_core(_N,_nt,NULL,_pers,_qs), n_sectors(_n_sectors)
		{
			map_funcs = (map_type*)_map_funcs;
			next_state_func = (next_state_type)_next_state;
			count_particles_func = (count_particles_type)_count_particles;
			op_func = (op_func_type)_op_func;
			ns_args = _ns_args;
			check_state_nosymm = (check_state_nosymm_type)_check_state_nosymm;
			csns_args = _csns_args;
		}

		~user_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			return (*map_funcs[n_map])(s,general_basis_core<I>::N,&sign);	
		}

		void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			map_type func = map_funcs[n_map];
			#pragma omp for schedule(static)
			for(npy_intp i=0;i<M;i++){
				int tempsign = 1;
				s[i] = (*func)(s[i],general_basis_core<I>::N,&tempsign);
				sign[i] *= (signed char)std::copysign(1,tempsign);

			}
		}

		std::vector<int> count_particles(const I s){
			std::vector<int> v(n_sectors);
			(*count_particles_func)(s,&v[0]);
			return v;
		}

		I inline next_state_pcon(const I s){
			return (*next_state_func)(s,(I)general_basis_core<I>::N,ns_args);
		}

		double check_state(I s){

			bool ns_check=true;
			if(check_state_nosymm!=0){
				ns_check = (*check_state_nosymm)(s,(I)general_basis_core<I>::N,csns_args);
			}			
			if(ns_check){
				return check_state_core_unrolled<I>(this,s,general_basis_core<I>::nt);
			}
			else{
				return std::numeric_limits<double>::quiet_NaN();
			}
		}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			op_results<I> res(m,r);
			for(int j=n_op-1;j>=0;j--){
				int err = (*op_func)(&res,opstr[j],indx[j],general_basis_core<I>::N);
				if(err!=0){
					return err;
				}
				if(std::abs(res.m)==0){
					res.r = s;
					break;
				}
			}
			m = res.m; r = res.r;
			return 0;
		}
};


}




#endif

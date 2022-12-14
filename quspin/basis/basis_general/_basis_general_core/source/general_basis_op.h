#ifndef _GENERAL_BASIS_OP_H
#define _GENERAL_BASIS_OP_H

#include <iostream>
#include <complex>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <boost/iterator/zip_iterator.hpp>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"
//#include "complex_ops.h"

namespace basis_general {


int general_inplace_op_get_switch_num(PyArray_Descr * dtype)
{
    int T = dtype->type_num;
    if(T==NPY_COMPLEX128){return 0;}
    else if(T==NPY_FLOAT64){return 1;}
    else if(T==NPY_COMPLEX64){return 2;}
    else if(T==NPY_FLOAT32){return 3;}
    else {return -1;}
}



template<bool transpose>
struct transpose_indices {
    inline static void call(npy_intp &i,npy_intp &j){};
};

template<>
struct transpose_indices<true> {
    inline static void call(npy_intp &i,npy_intp &j)
    {
        npy_intp k = j;
        j = i;
        i = k;
    }
};



template<bool conjugate>
struct conj
{
    inline static void call(std::complex<double> &z){}
};

template<>
struct conj<true>
{
    inline static void call(std::complex<double> &z){
        z = std::conj(z);
    }
};

template<class I,class P,
        bool full_basis,
        bool symmetries,
        bool bracket_basis>
struct get_index {};




template<class I,class P>
struct get_index<I,P,true,false,false>{
    inline static npy_intp call(general_basis_core<I,P> *B,
                                const int nt,
                                const I r,
                                const npy_intp Ns,
                                const I basis[],
                                const npy_intp basis_begin[],
                                const npy_intp basis_end[],
                                const int N_p,
                                      int g[],
                                      P &sign) 
    {
        return Ns - (npy_intp)r - 1;
    }

};

template<class I,class P>
struct get_index<I,P,false,false,false>
{
    inline static npy_intp call(general_basis_core<I,P> *B,
                                const int nt,
                                const I r,
                                const npy_intp Ns,
                                const I basis[],
                                const npy_intp basis_begin[],
                                const npy_intp basis_end[],
                                const int N_p,
                                      int g[],
                                      P &sign) 
    {
        return rep_position<npy_intp,I>(Ns,basis,r);
    }
};


template<class I,class P>
struct get_index<I,P,false,true,false> {
    inline static npy_intp call(general_basis_core<I,P> *B,
                                const int nt,
                                const I r,
                                const npy_intp Ns,
                                const I basis[],
                                const npy_intp basis_begin[],
                                const npy_intp basis_end[],
                                const int N_p,
                                      int g[],
                                      P &sign)
    {
        for(int k=0;k<nt;k++){
            g[k]=0;
        }
        I rr = B->ref_state(r,g,sign);
        return rep_position<npy_intp,I>(Ns,basis,rr);
    }
};


template<class I,class P>
struct get_index<I,P,false,true,true>
{
    inline static npy_intp call(general_basis_core<I,P> *B,
                                const int nt,
                                const I r,
                                const npy_intp Ns,
                                const I basis[],
                                const npy_intp basis_begin[],
                                const npy_intp basis_end[],
                                const int N_p,
                                      int g[],
                                      P &sign)
    {
        for(int k=0;k<nt;k++){
            g[k]=0;
        }
        I rr = B->ref_state(r,g,sign);
        npy_intp rr_prefix = B->get_prefix(rr,N_p);
        return rep_position<npy_intp,I>(basis_begin,basis_end,basis,rr_prefix,rr);
    }
};


template<class I,class P>
struct get_index<I,P,false,false,true>
{
    inline static npy_intp call(general_basis_core<I,P> *B,
                                const int nt,
                                const I r,
                                const npy_intp Ns,
                                const I basis[],
                                const npy_intp basis_begin[],
                                const npy_intp basis_end[],
                                const int N_p,
                                      int g[],
                                      P &sign)
    {
        npy_intp r_prefix = B->get_prefix(r,N_p);
        return rep_position<npy_intp,I>(basis_begin,basis_end,basis,r_prefix,r);
    }
};




template<class J,class P,bool symmetries>
struct scale_matrix_ele
{
    inline static void call(const int nt,
                            const npy_intp i,
                            const npy_intp j,
                            const P sign,
                            const J n[],
                            const int g[],
                            const double kk[],
                            std::complex<double> &m) {}
};

template<class J,class P>
struct scale_matrix_ele<J,P,true>
{
    inline static void call(const int nt,
                            const npy_intp i,
                            const npy_intp j,
                            const P sign,
                            const J n[],
                            const int g[],
                            const double kk[],
                            std::complex<double> &m) 
    {
        double q = 0;
        for(int k=0;k<nt;k++){
            q += kk[k]*g[k];                                
        }
        m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));
    }    
};








template<class I, class J, class K,class P=signed char,
            bool full_basis,bool symmetries,bool bracket_basis,
            bool transpose,bool conjugate>
int general_inplace_op_core(general_basis_core<I,P> *B,
                                  const int n_op,
                                  const char opstr[],
                                  const int indx[],
                                  const std::complex<double> A,
                                  const npy_intp Ns,
                                  const npy_intp nvecs,
                                  const I basis[],
                                  const J n[],
                                  const npy_intp basis_begin[],
                                  const npy_intp basis_end[],
                                  const int N_p,
                                  const K v_in[],
                                          K v_out[])
{
    int err = 0;
    #pragma omp parallel shared(err) firstprivate(A)
    {
        const int nt = B->get_nt();
        const int nthread = omp_get_num_threads();
        const npy_intp chunk = std::max(Ns/(1000*nthread),(npy_intp)1);

        std::complex<double> m;
        int g[__GENERAL_BASIS_CORE__max_nt];
        double kk[__GENERAL_BASIS_CORE__max_nt];

        for(int k=0;k<nt;k++)
            kk[k] = (2.0*M_PI*B->qs[k])/B->pers[k];

        #pragma omp for schedule(dynamic,chunk)
        for(npy_intp ii=0;ii<Ns;ii++){
            if(err != 0){
                continue;
            }
            
            npy_intp i = ii;

            const I s = basis[ii];
            I r = s;
            m = A;
            int local_err = B->op(r,m,n_op,opstr,indx);

            if(local_err == 0){
                P sign = 1;
                npy_intp j = i;
                bool not_diag = (r != s);
                if(not_diag){
                    j = get_index<I,P,full_basis,symmetries,bracket_basis>::call(B,nt,r,Ns,basis,basis_begin,basis_end,N_p,g,sign);
                }

                if(j >= 0){
                    if(symmetries && not_diag){
                        scale_matrix_ele<J,P,symmetries>::call(nt,i,j,sign,n,g,kk,m);                        
                    }

                    transpose_indices<transpose>::call(i,j);
                    conj<conjugate>::call(m);

                    const K * v_in_col  = v_in  + i * nvecs;
                          K * v_out_row = v_out + j * nvecs;

                    local_err = type_checks<K>(m);

                    if(local_err){
                        #pragma omp atomic write
                        err = local_err;
                    }

                    for(int k=0;k<nvecs;k++){
                        const std::complex<double> M = mul(v_in_col[k],m);
                        atomic_add(M,&v_out_row[k]);
                    }
                }
            }
            else if(err==0){
                #pragma omp atomic write
                err = local_err;
            }
        }
    }
    return err;
}

template<class I, class J, class K,class P=signed char>
int general_inplace_op(general_basis_core<I,P> *B,
                          const bool conjugate,
                          const bool transpose,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                          const std::complex<double> A,
                          const bool full_basis,
                          const npy_intp Ns,
                          const npy_intp nvecs,
                          const I basis[],
                          const J n[],
                          const npy_intp basis_begin[],
                          const npy_intp basis_end[],
                          const int N_p,
                          const K v_in[],
                                  K v_out[])
{
    int err = 0;
    const int nt = B->get_nt();
    if(full_basis){ // full_basis = true, symmetries = false, // bracket_basis = false
        if(transpose){ // transpose = true
            if(conjugate){ // conjugate = true
                return general_inplace_op_core<I,J,K,P,true,false,false,true,true>(
                    B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
            else{ // conjugate = false
                return general_inplace_op_core<I,J,K,P,true,false,false,true,false>(
                    B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
        }
        else{ // transpose = false
            if(conjugate){ // conjugate = true
                return general_inplace_op_core<I,J,K,P,true,false,false,false,true>(
                    B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
            else{ // conjugate = false
                return general_inplace_op_core<I,J,K,P,true,false,false,false,false>(
                    B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
        }
    }
    else if(nt>0){ // full_basis = false, symmetries = true 
        if(N_p>0){ // bracket_basis = true
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,true,true,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,true,true,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,true,false,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,true,false,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
        }
        else{ // bracket_basis = false
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,false,true,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,false,true,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,false,false,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,false,false,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }            
        }
    }
    else{ // full_basis = flase, symmetries = false
        if(N_p>0){ // bracket_basis = true
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,true,true,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,true,true,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,true,false,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,true,false,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
        }
        else{ // bracket_basis = false
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,false,true,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,false,true,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,false,false,true>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,false,false,false>(
                        B,n_op,opstr,indx,A,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }            
        }
    }
}

template<class I, class J,class P=signed char>
int general_inplace_op_impl(general_basis_core<I,P> *B,
                          const bool conjugate,
                          const bool transpose,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                                  void * A,
                          const bool full_basis,
                          const npy_intp Ns,
                          const npy_intp nvecs,
                          const I basis[],
                          const J n[],
                          const npy_intp basis_begin[],
                          const npy_intp basis_end[],
                          const int N_p,
                          PyArray_Descr * dtype,
                                  void * v_in,
                                  void * v_out)
{
    int type_num = dtype->type_num;
    switch (type_num)
    {
        case NPY_COMPLEX128:
            return general_inplace_op<I,J,std::complex<double>,P>(B,conjugate,transpose,n_op,opstr,indx,*(std::complex<double> *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const std::complex<double> *)v_in,(std::complex<double> *)v_out);
        case NPY_FLOAT64:
            return general_inplace_op<I,J,double,P>(B,conjugate,transpose,n_op,opstr,indx,*(std::complex<double> *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const double *)v_in,(double *)v_out);
        case NPY_COMPLEX64:
            return general_inplace_op<I,J,std::complex<float>,P>(B,conjugate,transpose,n_op,opstr,indx,*(std::complex<double> *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const std::complex<float> *)v_in,(std::complex<float> *)v_out);
        case NPY_FLOAT32:
            return general_inplace_op<I,J,float,P>(B,conjugate,transpose,n_op,opstr,indx,*(std::complex<double> *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const float *)v_in,(float *)v_out);
        default:
            return -2;
    }    
}



// template<class Tuple>
// struct compare_Tuple : std::binary_function<Tuple,Tuple,bool>
// {
//     bool operator()(const Tuple &a, const Tuple &b) const {
//         return (std::get<0>(a) == std::get<0>(b) ? std::get<1>(a) < std::get<1>(b) : std::get<0>(a) < std::get<0>(b));
//     }
// };


template<class T,class K>
using Tuple = boost::tuple<T&,K&,K&>;

template<class T>
struct nonzero : std::unary_function<T,bool>
{
    inline bool operator()(const T& tup) const {
        return equal_zero(boost::get<0>(tup));
    }
};


template<class I, class J, class K, class T,class P=signed char,
        bool full_basis,bool symmetries,bool bracket_basis>
std::pair<int,int> 
general_op_core(general_basis_core<I,P> *B,
                  const int n_op,
                  const char opstr[],
                  const int indx[],
                  const std::complex<double> A,
                  const npy_intp Ns,
                  const I basis[],
                  const J n[],
                  const npy_intp basis_begin[],
                  const npy_intp basis_end[],
                  const int N_p,
                          K row[],
                          K col[],
                          T M[]
                  )
{
    int err = 0, warn = 0;

    #pragma omp parallel firstprivate(N_p,Ns,A,n_op)
    {
        std::complex<double> m;
        const int N = B->get_N();
        const int nt = B->get_nt();
        const int nthread = omp_get_num_threads();
        const int threadn = omp_get_thread_num();
        const npy_intp chunk = (Ns+nthread-1)/nthread;
        const npy_intp begin = std::min(threadn*chunk,Ns);
        const npy_intp end   = std::min((threadn+1)*chunk,Ns);
        const npy_intp dyn_chunk = std::max(Ns/(1000*nthread),(npy_intp)1);

        int g[__GENERAL_BASIS_CORE__max_nt];
        double kk[__GENERAL_BASIS_CORE__max_nt];
        
        for(int k=0;k<nt;k++){
            kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
        }

        std::fill(M+begin,M+end,T(0));
        std::fill(row+begin,row+end,K(0));
        std::fill(col+begin,col+end,K(0));

        #pragma omp barrier

        #pragma omp for schedule(dynamic,dyn_chunk)
        for(npy_intp i=0;i<Ns;i++){

            I r = basis[i];
            const I s = r;
            m = A;
            int local_err = B->op(r,m,n_op,&opstr[0],&indx[0]);
            if(local_err == 0){
                P sign = 1;
                npy_intp j = i;
                bool not_diag = (r != s);
                if(not_diag){
                    j = get_index<I,P,full_basis,symmetries,bracket_basis>::call(B,nt,r,Ns,basis,basis_begin,basis_end,N_p,g,sign);
                }
    
                if(j >= 0){
                    if(symmetries && not_diag){
                        scale_matrix_ele<J,P,symmetries>::call(nt,i,j,sign,n,g,kk,m);                        
                    }

                    T me = 0;
                    int local_warn = type_checks(m,&me);

                    if(warn == 0 && local_warn != 0){
                        #pragma omp atomic write
                        warn = local_warn;
                    }

                    M[i]   = me;
                    col[i] = i;
                    row[i] = j;
                }
            }
            else if(err == 0){
                #pragma omp atomic write
                err = local_err;                    
            }
        }
    }

    return std::make_pair(err,warn);
}


template<class I, class J, class K,class T,class P=signed char>
std::pair<int,int> 
general_op(general_basis_core<I,P> *B,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                          const std::complex<double> A,
                          const bool full_basis,
                          const npy_intp Ns,
                          const I basis[],
                          const J n[],
                          const npy_intp basis_begin[],
                          const npy_intp basis_end[],
                          const int N_p,
                                  K row[],
                                  K col[],
                                  T M[]
                          )
{
    int err = 0;
    const int nt = B->get_nt();
    if(full_basis){ // full_basis = true, symmetries = false, // bracket_basis = false
        return general_op_core<I,J,K,T,P,true,false,false>(B,n_op,opstr,indx,A,Ns,basis,n,basis_begin,basis_end,N_p,row,col,M);
    }
    else if(nt>0){ // full_basis = false, symmetries = true 
        if(N_p>0){ // bracket_basis = true
            return general_op_core<I,J,K,T,P,false,true,true>(B,n_op,opstr,indx,A,Ns,basis,n,basis_begin,basis_end,N_p,row,col,M);
        }
        else{ // bracket_basis = false
            return general_op_core<I,J,K,T,P,false,true,false>(B,n_op,opstr,indx,A,Ns,basis,n,basis_begin,basis_end,N_p,row,col,M);
        }
    }
    else{ // full_basis = flase, symmetries = false
        if(N_p>0){ // bracket_basis = true
            return general_op_core<I,J,K,T,P,false,false,true>(B,n_op,opstr,indx,A,Ns,basis,n,basis_begin,basis_end,N_p,row,col,M);
        }
        else{ // bracket_basis = false
            return general_op_core<I,J,K,T,P,false,false,false>(B,n_op,opstr,indx,A,Ns,basis,n,basis_begin,basis_end,N_p,row,col,M);
        }
    }
}
















/*
typedef std::vector<std::complex<double>> J_list_type;
typedef std::vector<std::vector<int>> indx_list_type;
typedef std::vector<std::string> op_list_type;

template<class I, class J, class K,class P=signed char,
            bool full_basis,bool symmetries,bool bracket_basis,
            bool transpose,bool conjugate>
int general_inplace_op_core(general_basis_core<I,P> *B,
                                op_list_type op_list,
                                indx_list_type indx_list,
                                J_list_type J_list,
                                  const npy_intp Ns,
                                  const npy_intp nvecs,
                                  const I basis[],
                                  const J n[],
                                  const npy_intp basis_begin[],
                                  const npy_intp basis_end[],
                                  const int N_p,
                                  const K v_in[],
                                          K v_out[])
{
    int err = 0;
    const int nt = B->get_nt();
    const int nthread = omp_get_max_threads();
    const npy_intp chunk = std::max(Ns/(1000*nthread),(npy_intp)1);
    const int n_ops = J_list.size();

    std::vector<std::pair<npy_intp,K>> ME_vec(Ns);
    std::vector<std::complex<double>> tmp(nvecs*nthread);

    std::pair<npy_intp,K> * ME = &ME_vec[0];

    #pragma omp parallel shared(err,ME)
    {

        std::complex<double> m;
        K M = 0;
        int g[__GENERAL_BASIS_CORE__max_nt];
        double kk[__GENERAL_BASIS_CORE__max_nt];

        for(int k=0;k<nt;k++)
            kk[k] = (2.0*M_PI*B->qs[k])/B->pers[k];
        

        for(int o=0;o<n_ops;o++){
            if(err!=0){
                break;
            }

            const int n_op = indx_list[o].size();
            const char * opstr = op_list[o].c_str();
            const int * indx = indx_list[o].data();

            #pragma omp for schedule(dynamic,chunk)
            for(npy_intp ii=0;ii<Ns;ii++){
                if(err!=0){
                    continue;
                }

                int i = ii;
                const I s = basis[ii];
                I r = s;
                m = A;
                int local_err = B->op(r,m,n_op,opstr,indx);

                if(local_err == 0){
                    P sign = 1;
                    npy_intp j = i;
                    bool not_diag = (r != s);
                    if(not_diag){
                        j = get_index<I,P,full_basis,symmetries,bracket_basis>::call(B,nt,r,Ns,basis,basis_begin,basis_end,N_p,g,sign);
                    }

                    if(j >= 0){
                        if(not_diag){
                            scale_matrix_ele<J,P,symmetries>::call(nt,i,j,sign,n,g,kk,m);                        
                        }
                        transpose_indices<transpose>::call(i,j);
                        
                        local_err = type_checks(conj<conjugate>::call(m),&M);
                        ME[ii] = std::make_pair(j,M);

                        if(local_err){
                            #pragma omp atomic write
                            err = local_err;
                        }
                    }
                    else{
                        M = 0;
                        ME[ii] = std::make_pair(j,M);
                    }
                }
                else{
                    #pragma omp atomic write
                    err = local_err;
                }
            }




            if(err!=0){
                break;
            }



        }

    }
    return err;
}

template<class I, class J, class K,class P=signed char>
int general_inplace_op(general_basis_core<I,P> *B,
                          const bool conjugate,
                          const bool transpose,
                          const bool full_basis,
                                op_list_type op_list,
                                indx_list_type indx_list,
                                J_list_type J_list,
                          const npy_intp Ns,
                          const npy_intp nvecs,
                          const I basis[],
                          const J n[],
                          const npy_intp basis_begin[],
                          const npy_intp basis_end[],
                          const int N_p,
                          const K v_in[],
                                  K v_out[])
{
    int err = 0;
    const int nt = B->get_nt();
    if(full_basis){ // full_basis = true, symmetries = false, // bracket_basis = false
        if(transpose){ // transpose = true
            if(conjugate){ // conjugate = true
                return general_inplace_op_core<I,J,K,P,true,false,false,true,true>(
                    B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
            else{ // conjugate = false
                return general_inplace_op_core<I,J,K,P,true,false,false,true,false>(
                    B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
        }
        else{ // transpose = false
            if(conjugate){ // conjugate = true
                return general_inplace_op_core<I,J,K,P,true,false,false,false,true>(
                    B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
            else{ // conjugate = false
                return general_inplace_op_core<I,J,K,P,true,false,false,false,false>(
                    B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
            }
        }
    }
    else if(nt>0){ // full_basis = false, symmetries = true 
        if(N_p>0){ // bracket_basis = true
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,true,true,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,true,true,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,true,false,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,true,false,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
        }
        else{ // bracket_basis = false
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,false,true,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,false,true,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,true,false,false,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,true,false,false,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }            
        }
    }
    else{ // full_basis = flase, symmetries = false
        if(N_p>0){ // bracket_basis = true
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,true,true,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,true,true,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,true,false,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,true,false,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
        }
        else{ // bracket_basis = false
            if(transpose){ // transpose = true
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,false,true,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,false,true,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }
            else{ // transpose = false
                if(conjugate){ // conjugate = true
                    return general_inplace_op_core<I,J,K,P,false,false,false,false,true>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
                else{ // conjugate = false
                    return general_inplace_op_core<I,J,K,P,false,false,false,false,false>(
                        B,op_list,indx_list,J_list,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,v_in,v_out);
                }
            }            
        }
    }
}

template<class I, class J,class P=signed char>
int general_inplace_op_impl(general_basis_core<I,P> *B,
                          const bool conjugate,
                          const bool transpose,
                          const bool full_basis,
                                op_list_type op_list,
                                indx_list_type indx_list,
                                J_list_type J_list,
                          const npy_intp Ns,
                          const npy_intp nvecs,
                          const I basis[],
                          const J n[],
                          const npy_intp basis_begin[],
                          const npy_intp basis_end[],
                          const int N_p,
                          PyArray_Descr * dtype,
                                  void * v_in,
                                  void * v_out)
{
    int type_num = dtype->type_num;
    switch (type_num)
    {
        case NPY_COMPLEX128:
            return general_inplace_op<I,J,std::complex<double>,P>(B,conjugate,transpose,op_list,indx_list,J_list,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const std::complex<double> *)v_in,(std::complex<double> *)v_out);
        case NPY_FLOAT64:
            return general_inplace_op<I,J,double,P>(B,conjugate,transpose,op_list,indx_list,J_list,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const double *)v_in,(double *)v_out);
        case NPY_COMPLEX64:
            return general_inplace_op<I,J,std::complex<float>,P>(B,conjugate,transpose,op_list,indx_list,J_list,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const std::complex<float> *)v_in,(std::complex<float> *)v_out);
        case NPY_FLOAT32:
            return general_inplace_op<I,J,float,P>(B,conjugate,transpose,op_list,indx_list,J_list,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const float *)v_in,(float *)v_out);
        default:
            return -2;
    }    
}

*/










/*

template<class I, class J, class K, class T,class P=signed char>
int general_op(general_basis_core<I,P> *B,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                          const std::complex<double> A,
                          const bool full_basis,
                          const npy_intp Ns,
                          const I basis[],
                          const J n[],
                                  K row[],
                                  K col[],
                                  T M[]
                          )
{
    int err = 0;
    #pragma omp parallel 
    {
        const int nt = B->get_nt();
        const int N = B->get_N();
        const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
        int g[__GENERAL_BASIS_CORE__max_nt];
        double kk[__GENERAL_BASIS_CORE__max_nt];

        for(int k=0;k<nt;k++)
            kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];


        #pragma omp for schedule(dynamic,chunk)
        for(npy_intp i=0;i<Ns;i++){
            if(err != 0){
                continue;
            }

            I r = basis[i];
            std::complex<double> m = A;
            int local_err = B->op(r,m,n_op,opstr,indx);

            if(local_err == 0){
                P sign = 1;

                for(int k=0;k<nt;k++){
                    g[k]=0;
                }

                K j = i;
                if(r != basis[i]){
                    I rr = B->ref_state(r,g,sign);
                    if(full_basis){
                        j = Ns - (npy_intp)rr - 1;
                    }
                    else{
                        j = rep_position(Ns,basis,rr);
                    }
                    
                }
                if(j >= 0){
                    double q = 0;
                    for(int k=0;k<nt;k++){
                        q += kk[k]*g[k];
                    }
                    m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));
                    
                    local_err = type_checks(m,&M[i]);
                    col[i]=i;
                    row[i]=j;
                }
                else{
                    col[i] = i;
                    row[i] = i;
                    M[i] = std::numeric_limits<T>::quiet_NaN();
                }
            }
            else{
                #pragma omp critical
                err = local_err;
            }
        }

        
    }
    return err;
}


template<class I, class J, class K, class T,class P=signed char>
int general_op(general_basis_core<I,P> *B,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                          const std::complex<double> A,
                          const bool full_basis,
                          const npy_intp Ns,
                          const I basis[],
                          const J n[],
                          const npy_intp basis_begin[],
                          const npy_intp basis_end[],
                          const int N_p,
                                  K row[],
                                  K col[],
                                  T M[]
                          )
{
    int err = 0;

    #pragma omp parallel 
    {
        const int N = B->get_N();
        const int nt = B->get_nt();

        const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
        int g[__GENERAL_BASIS_CORE__max_nt];
        double kk[__GENERAL_BASIS_CORE__max_nt];

        for(int k=0;k<nt;k++)
            kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];


        #pragma omp for schedule(dynamic,chunk)
        for(npy_intp i=0;i<Ns;i++){
            if(err != 0){
                continue;
            }

            I r = basis[i];
            const I s = r;
            std::complex<double> m = A;
            int local_err = B->op(r,m,n_op,opstr,indx);

            if(local_err == 0){
                P sign = 1;

                for(int k=0;k<nt;k++){
                    g[k]=0;
                }

                K j = i;
                const bool not_diag = r != s;
                if(r != basis[i]){
                    I rr = B->ref_state(r,g,sign);
                    if(full_basis){
                        j = Ns - (K)rr - 1;
                    }
                    else{
                        npy_intp rr_prefix = B->get_prefix(rr,N_p);
                        j = rep_position<K,I>(basis_begin,basis_end,basis,rr_prefix,rr);
                    }
                    
                }
    
                if(j >= 0){
                    if(not_diag){
                        double q = 0;
                        for(int k=0;k<nt;k++){
                            q += kk[k]*g[k];
                        }
                        m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));                        
                    }

                    
                    local_err = type_checks(m,&M[i]);
                    col[i] = i;
                    row[i] = j;
                }
                else{
                    col[i] = i;
                    row[i] = i;
                    M[i] = std::numeric_limits<double>::quiet_NaN();
                }
            }
            else{
                #pragma omp critical
                err = local_err;
            }
        }

        
    }
    return err;
}
*/



template<class I1,class J1,class I2,class J2,class K,class P=signed char>
int general_op_shift_sectors(general_basis_core<I1,P> *B_out,
                             const int n_op,
                             const char opstr[],
                             const int indx[],
                             const std::complex<double> A,
                             const npy_intp Ns_out,
                             const I1 basis_out[],
                             const J1 n_out[],
                             const npy_intp Ns_in,
                             const I2 basis_in[],
                             const J2 n_in[],
                             const npy_intp nvecs,
                             const K v_in[],
                                   K v_out[])
{
    int err = 0;
    #pragma omp parallel firstprivate(A)
    {
        const int nt = B_out->get_nt();
        const int nthread = omp_get_num_threads();
        const npy_intp dyn_chunk = std::max(Ns_in/(1000*nthread),(npy_intp)1);
        int g[__GENERAL_BASIS_CORE__max_nt];
        double kk[__GENERAL_BASIS_CORE__max_nt];

        for(int k=0;k<nt;k++){
            kk[k] = (2.0*M_PI*B_out->qs[k])/B_out->pers[k];
        }

        #pragma omp for schedule(dynamic,dyn_chunk)
        for(npy_intp i=0;i<Ns_in;i++){

            std::complex<double> m = A;
            const I1 s = (I1)basis_in[i];
            I1 r = s;
            
            int local_err = B_out->op(r,m,n_op,opstr,indx);

            if(local_err == 0){
                P sign = 1;

                if(r != s){ // off-diagonal matrix element
                    for(int k=0;k<nt;k++){g[k]=0;}
                    r = B_out->ref_state(r,g,sign);
                }
                
                npy_intp j = rep_position<npy_intp,I1>(Ns_out,basis_out,r);

                if(j>=0){ // ref_state is a representative

                    if(r != s){
                        double q = 0;
                        for(int k=0;k<nt;k++){
                            q += kk[k]*g[k];                              
                        }
                        m *= (double)sign * std::exp(std::complex<double>(0,-q));    
                    }
                    
                    m *= std::sqrt(double(n_out[j])/double(n_in[i]));
                    const K * v_in_col  = v_in  + i * nvecs;
                          K * v_out_row = v_out + j * nvecs;

                    local_err = type_checks<K>(m);

                    if(local_err && err==0){
                        #pragma omp atomic write
                        err = local_err;                       
                    }

                    for(int k=0;k<nvecs;k++){
                        const std::complex<double> M = mul(v_in_col[k],m);
                        atomic_add(M,&v_out_row[k]);
                    }
                }
            }
            else if(err==0){
                #pragma omp atomic write
                err = local_err;                    
            }
        }

        
    }
    return err;
}





template<class I, class T, class P=signed char>
int general_op_bra_ket(general_basis_core<I,P> *B,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                          const std::complex<double> A,
                          const npy_intp Ns,
                          const I ket[], // col
                                  I bra[], // row
                                  T M[]
                          )
{
    int err = 0;
    #pragma omp parallel
    {
        const int nt = B->get_nt();
        int g[__GENERAL_BASIS_CORE__max_nt];
            
        #pragma omp for schedule(static)
        for(npy_intp i=0;i<Ns;i++){
            if(err != 0){
                continue;
            }

            std::complex<double> m = A;
            const I s = ket[i];
            I r = ket[i];
            
            int local_err = B->op(r,m,n_op,opstr,indx);

            if(local_err == 0){
                P sign = 1;

                if(r != s){ // off-diagonal matrix element
                    r = B->ref_state(r,g,sign);
                    // use check_state to determine if state is a representative (same routine as in make-general_basis)
                    double norm_r = B->check_state(r);
                    npy_intp int_norm = norm_r;

                    if(!check_nan(norm_r) && int_norm > 0){ // ref_state is a representative

                        for(int k=0;k<nt;k++){
                            double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
                            m *= std::exp(std::complex<double>(0,-q));
                        }

                        double norm_s = B->check_state(s);
                        m *= sign * std::sqrt(norm_r/norm_s);

                        local_err = type_checks(m,&M[i]); // assigns value to M[i]
                        bra[i] = r;
                    }
                    else{ // ref state in different particle number sector
                        M[i] = std::numeric_limits<T>::quiet_NaN();
                        bra[i] = s;
                    }
                }
                else{ // diagonal matrix element
                    m *= sign;
                    local_err = type_checks(m,&M[i]); // assigns value to M[i]
                    bra[i] = s;
                }
                
                
            }
            else{
                #pragma omp critical
                err = local_err;
            }
        }

        
    }
    return err;
}






template<class I, class T, class P=signed char>
int general_op_bra_ket_pcon(general_basis_core<I,P> *B,
                          const int n_op,
                          const char opstr[],
                          const int indx[],
                          const std::complex<double> A,
                          const npy_intp Ns,
                          const std::set<std::vector<int>> &Np_set, // array with particle conserving sectors
                          const    I ket[], // col
                                  I bra[], // row
                                  T M[]
                          )
{
    int err = 0;

    #pragma omp parallel
    {
        const std::set<std::vector<int>> Np_set_local = Np_set;
        const int nt = B->get_nt();
        int g[__GENERAL_BASIS_CORE__max_nt];
        
        #pragma omp for schedule(static) 
        for(npy_intp i=0;i<Ns;i++){
            if(err != 0){
                continue;
            }

            std::complex<double> m = A;
            const I s = ket[i];
            I r = ket[i];
            
            int local_err = B->op(r,m,n_op,opstr,indx);

            if(local_err == 0){
                P sign = 1;


                if(r != s){ // off-diagonal matrix element
                    r = B->ref_state(r,g,sign);

                    bool pcon_bool = B->check_pcon(r,Np_set_local);

                    if(pcon_bool){ // reference state within same particle-number sector(s)

                        // use check_state to determine if state is a representative (same routine as in make-general_basis)
                        double norm_r = B->check_state(r);

                        if(!check_nan(norm_r) && norm_r > 0){ // ref_state is a representative

                            for(int k=0;k<nt;k++){
                                double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
                                m *= std::exp(std::complex<double>(0,-q));
                            }

                            double norm_s = B->check_state(s);
                            m *= sign * std::sqrt(norm_r/norm_s);

                            local_err = type_checks(m,&M[i]); // assigns value to M[i]
                            bra[i] = r;

                        }
                        else{ // ref_state not a representative
                            M[i] = std::numeric_limits<T>::quiet_NaN();
                            bra[i] = s;
                        }

                    }
                    else{ // ref state in different particle number sector
                        M[i] = std::numeric_limits<T>::quiet_NaN();
                        bra[i] = s;
                    }

                    
                }
                else{ // diagonal matrix element

                    for(int k=0;k<nt;k++){
                        double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
                        m *= std::exp(std::complex<double>(0,-q));
                    }

                    m *= sign;

                    local_err = type_checks(m,&M[i]); // assigns value to M[i]
                    bra[i] = s;
                }
                
                
            }
            else{
                #pragma omp critical
                err = local_err;
            }
        }

        
    }
    return err;
}


}

#endif

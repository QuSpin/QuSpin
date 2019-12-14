#ifndef __CSR_H
#define __CSR_H

#include "complex_ops.h"
#include "utils.h"
#include <iostream>
#include <iomanip>

void write(const float& a){
    std::cout << a;
}

void write(const double& a){
    std::cout << a;
}

void write(const npy_cfloat_wrapper& a){
    std::cout << a.real << std::setw(20) << a.imag;
}

void write(const npy_cdouble_wrapper& a){
    std::cout << a.real << std::setw(20) << a.imag;
}

template<typename I, typename T1, typename T2,typename T3>
void csr_matvec_noomp_contig(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T2 a,
                const T3 x[],
                      T3 y[])
{
    // const T3 a_cast = T3(a);
    if(overwrite_y){
        for(I k = 0; k<n_row; k++){
            T3 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj]];
            }
            y[k] = a * sum;
        }
    }else{
        for(I k = 0; k<n_row; k++){
            T3 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj]];
            }

            y[k] += a * sum;
        }
    }
}

template<typename I, typename T1, typename T2,typename T3>
void csr_matvec_noomp_strided(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T2 a,
                const npy_intp x_stride,
                const T3 x[],
                const npy_intp y_stride,
                      T3 y[])
{
    // const T3 a_cast = T3(a);
    if(overwrite_y){
        for(I k = 0; k<n_row; k++){
            T3 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj] * x_stride];
            }
            y[k * y_stride] = a * sum;
        }
    }else{
        for(I k = 0; k<n_row; k++){
            T3 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj] * x_stride];
            }
            y[k * y_stride] += a * sum;
        }
    }
}



template<typename I, typename T1, typename T2,typename T3>
void csr_matvecs_noomp_strided(const bool overwrite_y,
                        const I n_row,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
    if(overwrite_y){
        for(npy_intp i = 0; i < n_row; i++){
            for(npy_intp j = 0; j < n_vecs; j++){
                y[i * y_stride_row + j * y_stride_col] = 0;
            }
        }
    }

    if(y_stride_col < y_stride_row){
        for(I k = 0; k<n_row; k++){
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                const T3 ax = a * Ax[jj];
                const T3 * x_row = x + x_stride_row *  Aj[jj];
                axpy_strided(n_vecs, ax, x_stride_col, x_row, y_stride_col, y);
            }
            y += y_stride_row;
        }
    }
    else{
        for(I m=0;m<n_vecs; m++){
            for(I k = 0; k<n_row; k++){
                for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                    const npy_intp ii = x_stride_row *  Aj[jj];
                    (*y) += (a * Ax[jj]) * x[ii];
                }
                y += y_stride_row;
            }
            x += x_stride_col;
        }
    }
}





#if defined(_OPENMP)
#include "csrmv_merge.h"
#include "openmp.h"

template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvec_omp_contig(const bool overwrite_y,
                        const I n_row,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const T3 x[],
                              T3 y[])
{
	const int nthread = omp_get_max_threads();
	std::vector<I> rco_vec(nthread);
	std::vector<T3> vco_vec(nthread);
	I * rco = &rco_vec[0];
	T3 * vco = &vco_vec[0];
	#pragma omp parallel shared(Ap,Aj,Ax,x,rco,vco,y) firstprivate(overwrite_y,n_row)
	{
		csrmv_merge(overwrite_y,n_row,Ap,Aj,Ax,a,x,rco,vco,y);
	}
}

template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvec_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride,
                        const T3 x[],
                        const npy_intp y_stride,
                              T3 y[])
{
	const int nthread = omp_get_max_threads();

	std::vector<I> rco_vec(nthread);
	std::vector<T3> vco_vec(nthread);
	I * rco = &rco_vec[0];
	T3 * vco = &vco_vec[0];
	#pragma omp parallel
	{
		csrmv_merge_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,rco,vco,y_stride,y);
	}
}


template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
    csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}

#else

template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvec_omp_contig(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T2 a,
                const T3 x[],
                      T3 y[])
{
    csr_matvec_noomp_contig(overwrite_y,n_row,Ap,Aj,Ax,a,x,y);
}

template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvec_omp_strided(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T2 a,
                const npy_intp x_stride,
                const T3 x[],
                const npy_intp y_stride,
                      T3 y[])
{
    csr_matvec_noomp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,y_stride,y);
}


template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
    csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}

#endif



 // when openmp is not being used omp and noomp versions are identical

template<typename I, typename T1, typename T2,typename T3>
void csr_matvec_noomp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_byte,
                        const T3 x[],
                        const npy_intp y_stride_byte,
                              T3 y[])
{
	const npy_intp y_stride = y_stride_byte/sizeof(T3);
	const npy_intp x_stride = x_stride_byte/sizeof(T3);
    
    if(y_stride == 1){
        if(x_stride == 1){
            csr_matvec_noomp_contig(overwrite_y,n_row,Ap,Aj,Ax,a,x,y);
        }
        else{
            csr_matvec_noomp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,1,y);
        }   
    }
    else{
        if(x_stride == 1){
            csr_matvec_noomp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,1,x,y_stride,y);
        }
        else{
            csr_matvec_noomp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,y_stride,y);
        }   
    }
}

template<typename I, typename T1, typename T2,typename T3>
void csr_matvec_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_byte,
                        const T3 x[],
                        const npy_intp y_stride_byte,
                              T3 y[])
{
	const npy_intp y_stride = y_stride_byte/sizeof(T3);
	const npy_intp x_stride = x_stride_byte/sizeof(T3);

    if(y_stride == 1){
        if(x_stride == 1){
            csr_matvec_omp_contig(overwrite_y,n_row,Ap,Aj,Ax,a,x,y);
        }
        else{
            csr_matvec_omp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,1,y);
        }   
    }
    else{
        if(x_stride == 1){
            csr_matvec_omp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,1,x,y_stride,y);
        }
        else{
            csr_matvec_omp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,y_stride,y);
        }   
    }
}

template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvecs_noomp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row_byte,
                        const npy_intp x_stride_col_byte,
                        const T3 x[],
                        const npy_intp y_stride_row_byte,
                        const npy_intp y_stride_col_byte,
                              T3 y[])
{
    const npy_intp y_stride_row = y_stride_row_byte/sizeof(T3);
    const npy_intp y_stride_col = y_stride_col_byte/sizeof(T3);
    const npy_intp x_stride_row = x_stride_row_byte/sizeof(T3);
    const npy_intp x_stride_col = x_stride_col_byte/sizeof(T3);

    if(y_stride_col==1){
        if(x_stride_col==1){
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,1,y);
        }
    }
    else if(y_stride_row==1){
        if(x_stride_col==1){
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,1,y_stride_col,y);
        }
        else if(x_stride_row==1){
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,1,y_stride_col,y);
        }
        else{
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
}

template<typename I, typename T1, typename T2,typename T3>
inline void csr_matvecs_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row_byte,
                        const npy_intp x_stride_col_byte,
                        const T3 x[],
                        const npy_intp y_stride_row_byte,
                        const npy_intp y_stride_col_byte,
                              T3 y[])
{
    const npy_intp y_stride_row = y_stride_row_byte/sizeof(T3);
    const npy_intp y_stride_col = y_stride_col_byte/sizeof(T3);
    const npy_intp x_stride_row = x_stride_row_byte/sizeof(T3);
    const npy_intp x_stride_col = x_stride_col_byte/sizeof(T3);

    if(y_stride_col==1){
        if(x_stride_col==1){
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,1,y);
        }
    }
    else if(y_stride_row==1){
        if(x_stride_col==1){
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,1,y_stride_col,y);
        }
        else if(x_stride_row==1){
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,1,y_stride_col,y);
        }
        else{
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
}

#endif	
#ifndef __CSR_H
#define __CSR_H

#include "complex_ops.h"
#include "utils.h"



template<typename I, typename T1,typename T2>
void csr_matvec_noomp_contig(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T1 a,
                const T2 x[],
                      T2 y[])
{
    const T2 a_cast = T2(a);
    if(overwrite_y){
        for(I k = 0; k<n_row; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj]];
            }
            y[k] = a_cast * sum;
        }
    }else{
        for(I k = 0; k<n_row; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj]];
            }
            y[k] += a_cast * sum;
        }
    }
}

template<typename I, typename T1,typename T2>
void csr_matvec_noomp_strided(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T1 a,
                const npy_intp x_stride,
                const T2 x[],
                const npy_intp y_stride,
                      T2 y[])
{
    const T2 a_cast = T2(a);
    if(overwrite_y){
        for(I k = 0; k<n_row; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj] * x_stride];
            }
            y[k * y_stride] = a_cast * sum;
        }
    }else{
        for(I k = 0; k<n_row; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += Ax[jj] * x[Aj[jj] * x_stride];
            }
            y[k * y_stride] += a_cast * sum;
        }
    }
}



template<typename I, typename T1,typename T2>
void csr_matvecs_noomp_strided(const bool overwrite_y,
                        const I n_row,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T2 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T2 y[])
{
    
    if(overwrite_y){
        const npy_intp m = (npy_intp)n_row * n_vecs;

        for(npy_intp i = 0; i < m; i++){
            y[i] = T2(0);
        }
    }


    if(y_stride_col < y_stride_row){
        for(I k = 0; k<n_row; k++){
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                const T2 ax = T2(a * Ax[jj]);
                const T2 * x_row = x + x_stride_row *  Aj[jj];
                axpy_strided((npy_intp)n_vecs, ax, x_stride_col, x_row, y_stride_col, y);
            }
            y += y_stride_row;
        }
    }
    else{
        for(I m=0;m<n_vecs; m++){
            for(I k = 0; k<n_row; k++){
                for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                    const npy_intp ii = x_stride_row *  Aj[jj];
                    (*y) += T2(a * Ax[jj]) * x[ii];
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

template<typename I, typename T1,typename T2>
inline void csr_matvec_omp_contig(const bool overwrite_y,
                        const I n_row,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const T2 x[],
                              T2 y[])
{
	const int nthread = omp_get_max_threads();
	std::vector<I> rco_vec(nthread);
	std::vector<T2> vco_vec(nthread);
	I * rco = &rco_vec[0];
	T2 * vco = &vco_vec[0];
	#pragma omp parallel shared(Ap,Aj,Ax,x,rco,vco,y) firstprivate(overwrite_y,n_row)
	{
		csrmv_merge(overwrite_y,n_row,Ap,Aj,Ax,a,x,rco,vco,y);
	}
}

template<typename I, typename T1,typename T2>
inline void csr_matvec_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride,
                        const T2 x[],
                        const npy_intp y_stride,
                              T2 y[])
{
	const int nthread = omp_get_max_threads();

	std::vector<I> rco_vec(nthread);
	std::vector<T2> vco_vec(nthread);
	I * rco = &rco_vec[0];
	T2 * vco = &vco_vec[0];
	#pragma omp parallel
	{
		csrmv_merge_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,rco,vco,y_stride,y);
	}
}


template<typename I, typename T1,typename T2>
inline void csr_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T2 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T2 y[])
{
    csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}

#else

template<typename I, typename T1,typename T2>
inline void csr_matvec_omp_contig(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T1 a,
                const T2 x[],
                      T2 y[])
{
    csr_matvec_noomp_contig(overwrite_y,n_row,Ap,Aj,Ax,a,x,y);
}

template<typename I, typename T1,typename T2>
inline void csr_matvec_omp_strided(const bool overwrite_y,
                const I n_row,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T1 a,
                const npy_intp x_stride,
                const T2 x[],
                const npy_intp y_stride,
                      T2 y[])
{
    csr_matvec_noomp_strided(overwrite_y,n_row,Ap,Aj,Ax,a,x_stride,x,y_stride,y);
}


template<typename I, typename T1,typename T2>
inline void csr_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T2 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T2 y[])
{
    csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}

#endif



 // when openmp is not being used omp and noomp versions are identical

template<typename I, typename T1,typename T2>
void csr_matvec_noomp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_byte,
                        const T2 x[],
                        const npy_intp y_stride_byte,
                              T2 y[])
{
	const npy_intp y_stride = y_stride_byte/sizeof(T2);
	const npy_intp x_stride = x_stride_byte/sizeof(T2);

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

template<typename I, typename T1,typename T2>
void csr_matvec_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_byte,
                        const T2 x[],
                        const npy_intp y_stride_byte,
                              T2 y[])
{
	const npy_intp y_stride = y_stride_byte/sizeof(T2);
	const npy_intp x_stride = x_stride_byte/sizeof(T2);

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

template<typename I, typename T1,typename T2>
inline void csr_matvecs_noomp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_row_byte,
                        const npy_intp x_stride_col_byte,
                        const T2 x[],
                        const npy_intp y_stride_row_byte,
                        const npy_intp y_stride_col_byte,
                              T2 y[])
{
    const npy_intp y_stride_row = y_stride_row_byte/sizeof(T2);
    const npy_intp y_stride_col = y_stride_col_byte/sizeof(T2);
    const npy_intp x_stride_row = x_stride_row_byte/sizeof(T2);
    const npy_intp x_stride_col = x_stride_col_byte/sizeof(T2);

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
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        csr_matvecs_noomp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
    

}

template<typename I, typename T1,typename T2>
inline void csr_matvecs_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride_row_byte,
                        const npy_intp x_stride_col_byte,
                        const T2 x[],
                        const npy_intp y_stride_row_byte,
                        const npy_intp y_stride_col_byte,
                              T2 y[])
{
    const npy_intp y_stride_row = y_stride_row_byte/sizeof(T2);
    const npy_intp y_stride_col = y_stride_col_byte/sizeof(T2);
    const npy_intp x_stride_row = x_stride_row_byte/sizeof(T2);
    const npy_intp x_stride_col = x_stride_col_byte/sizeof(T2);

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
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            csr_matvecs_omp_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,y_stride_row,1,y);
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
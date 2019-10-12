#ifndef __MATVEC_H__
#define __MATVEC_H__

#include <complex>
#include <algorithm>
#include "numpy/ndarraytypes.h"
#include "openmp.h"

#if defined(_OPENMP)
#include "csrmv_merge.h"

template<typename I, typename T1,typename T2>
void inline csr_matvec_contig(const bool overwrite_y,
                        const I n,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const T2 x[],
                              I rco[],
                              T2 vco[],
                              T2 y[])
{
    csrmv_merge(overwrite_y,n,Ap,Aj,Ax,a,x,rco,vco,y);
}

template<typename I, typename T1,typename T2>
void inline csr_matvec_strided(const bool overwrite_y,
                        const I n,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride,
                        const T2 x[],
                              I rco[],
                              T2 vco[],
                        const npy_intp y_stride,
                              T2 y[])
{
    csrmv_merge_strided(overwrite_y,n,Ap,Aj,Ax,a,x_stride,x,rco,vco,y_stride,y);
}

template <typename I, typename T1, typename T2>
void dia_matvec_contig(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
                const I offsets[], 
                const T1 diags[], 
                const T1 a,
                const T2 x[],
                      T2 y[])
{

    if(overwrite_y){
        #pragma omp for schedule(static)
        for(I n=0;n<n_row;n++){
            y[n] = 0; 
        }
    }

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process
        const T1 * diag = diags + i*L + j_start;
        const T2 * x_row = x + j_start;
              T2 * y_row = y + i_start;

        #pragma omp for schedule(static)
        for(I n=0;n<N;n++){
            y_row[n] += (T2)(a * diag[n]) * x_row[n]; 
        }
    }
}

template <typename I, typename T1, typename T2>
void dia_matvec_strided(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
                const I offsets[], 
                const T1 diags[], 
                const T1 a,
                const npy_intp x_stride,
                const T2 x[],
                const npy_intp y_stride,
                      T2 y[])
{

    if(overwrite_y){
        #pragma omp for schedule(static)
        for(I n=0;n<n_row;n++){
            y[n * y_stride] = 0; 
        }
    }

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process
        const T1 * diag = diags + i*L + j_start;
        const T2 * x_row = x + j_start * x_stride;
              T2 * y_row = y + i_start * y_stride;

        #pragma omp for schedule(static)
        for(I n=0;n<N;n++){
            y_row[n * y_stride] += (T2)(a * diag[n]) * x_row[n * x_stride]; 
        }
    }
}



template<typename I, typename T1,typename T2>
void csc_matvec_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T1 a,
                        const T2 x[],
                              T2 y[])
{
    const int nthread = omp_get_num_threads();
    const I chunk = std::max((I)1,n_row/(100*nthread));
    if(overwrite_y){
        #pragma omp for schedule(static)
        for(I j = 0; j < n_row; j++){
            y[j] = 0;
        }
    }
    
    #pragma omp for schedule(dynamic,chunk)
    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            const I i = Ai[ii];
            const T2 aa = (T2)(a * Ax[ii]) * x[j];
            atomic_add(y[i],aa);
        }
    }
}
 

template<typename I, typename T1,typename T2>
void csc_matvec_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride,
                        const T2 x[],
                        const npy_intp y_stride,
                              T2 y[])
{
    const int nthread = omp_get_num_threads();
    const I chunk = std::max((I)1,n_row/(100*nthread));
    if(overwrite_y){
        #pragma omp for schedule(static)
        for(I j = 0; j < n_row; j++){
            y[j * y_stride] = 0;
        }
    }
    
    #pragma omp for schedule(dynamic,chunk)
    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            const I i = Ai[ii];
            const T2 aa = (T2)(a * Ax[ii]) * x[j * x_stride];
            atomic_add(y[i * y_stride],aa);
        }
    }
}



#else

template<typename I, typename T1,typename T2>
void csr_matvec_contig(const bool overwrite_y,
                const I n,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T1 a,
                const T2 x[],
                      I rco[],
                      T2 vco[],
                      T2 y[])
{
    const T2 a_cast = a;
    if(overwrite_y){
        for(I k = 0; k<n; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += (T2)Ax[jj] * x[Aj[jj]];
            }
            y[k] = a_cast * sum;
        }
    }else{
        for(I k = 0; k<n; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += (T2)Ax[jj] * x[Aj[jj]];
            }
            y[k] += a_cast * sum;
        }
    }
}

template<typename I, typename T1,typename T2>
void csr_matvec_strided(const bool overwrite_y,
                const I n,
                const I Ap[],
                const I Aj[],
                const T1 Ax[],
                const T1 a,
                const npy_intp x_stride,
                const T2 x[],
                      I rco[],
                      T2 vco[],
                const npy_intp y_stride,
                      T2 y[])
{
    const T2 a_cast = a;
    if(overwrite_y){
        for(I k = 0; k<n; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += (T2)Ax[jj] * x[Aj[jj] * x_stride];
            }
            y[k * y_stride] = a_cast * sum;
        }
    }else{
        for(I k = 0; k<n; k++){
            T2 sum = 0;
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                sum += (T2)Ax[jj] * x[Aj[jj] * x_stride];
            }
            y[k * y_stride] += a_cast * sum;
        }
    }
}

template <typename I, typename T1, typename T2>
void dia_matvec_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T1 a,
                        const T2 x[],
                              T2 y[])
{

    if(overwrite_y){
        for(I i = 0; i < n_row; i++){
            y[i] = 0;
        }
    }

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T2 * x_row = x + j_start;
              T2 * y_row = y + i_start;

        for(I n = 0; n < N; n++){
            y_row[n] += (T2)(a * diag[n]) * x_row[n]; 
        }
    }
}

template <typename I, typename T1, typename T2>
void dia_matvec_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T1 a,
                        const npy_intp x_stride,
                        const T2 x[],
                        const npy_intp y_stride,
                              T2 y[])
{

    if(overwrite_y){
        for(I i = 0; i < n_row; i++){
            y[i] = 0;
        }
    }

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T2 * x_row = x + j_start * x_stride;
              T2 * y_row = y + i_start * y_stride;

        for(I n = 0; n < N; n++){
            y_row[n * y_stride] += (T2)(a * diag[n]) * x_row[n * x_stride]; 
        }
    }
}

template<typename I, typename T1,typename T2>
void csc_matvec_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T1 a,
                        const T2 x[],
                              T2 y[])
{
    if(overwrite_y){
        for(I j = 0; j < n_row; j++){
            y[j] = 0;
        }
    }

    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            const I i = Ai[ii];
            y[i] += (T2)(a * Ax[ii]) * x[j];
        }
    }
}

template<typename I, typename T1,typename T2>
void csc_matvec_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride,
                        const T2 x[],
                        const npy_intp y_stride,
                              T2 y[])
{
    if(overwrite_y){
        for(I j = 0; j < n_row; j++){
            y[j] = 0;
        }
    }

    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            const I i = Ai[ii];
            y[i * y_stride] += (T2)(a * Ax[ii]) * x[j * x_stride];
        }
    }
}



#endif











template<typename I, typename T1,typename T2>
void csr_matvec(const bool overwrite_y,
                        const I n,
                        const I Ap[],
                        const I Aj[],
                        const T1 Ax[],
                        const T1 a,
                        const npy_intp x_stride,
                        const T2 x[],
                              I rco[],
                              T2 vco[],
                        const npy_intp y_stride,
                              T2 y[])
{
    if(y_stride == 1 && x_stride == 1){
        csr_matvec_contig(overwrite_y,n,Ap,Aj,Ax,a,x,rco,vco,y);    
    }
    else{
        csr_matvec_strided(overwrite_y,n,Ap,Aj,Ax,a,x_stride,x,rco,vco,y_stride,y);
    }
}


template <typename I, typename T1, typename T2>
void dia_matvec(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
                const I offsets[], 
                const T1 diags[], 
                const T1 a,
                const npy_intp x_stride,
                const T2 x[],
                const npy_intp y_stride,
                      T2 y[])
{
    if(y_stride == 1 && x_stride == 1){
        dia_matvec_contig(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x,y);
    }
    else{
        dia_matvec_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,y_stride,y);
    }    
}



template<typename I, typename T1,typename T2>
void csc_matvec(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I Ap[],
                const I Ai[],
                const T1 Ax[],
                const T1 a,
                const npy_intp x_stride,
                const T2 x[],
                const npy_intp y_stride,
                      T2 y[])
{
    if(y_stride == 1 && x_stride == 1){
        csc_matvec_contig(overwrite_y,n_row,n_col,Ap,Ai,Ax,a,x,y);
    }
    else{
        csc_matvec_strided(overwrite_y,n_row,n_col,Ap,Ai,Ax,a,x_stride,x,y_stride,y);
    }
}





#endif
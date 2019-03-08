#ifndef __DIA_H
#define __DIA_H 


#include <complex>
#include <algorithm>
#include "numpy/ndarraytypes.h"




template <typename I, typename T1, typename T2>
void dia_matvec_noomp_contig(const bool overwrite_y,
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
            y_row[n] += T2(a * diag[n]) * x_row[n]; 
        }
    }
}

template <typename I, typename T1, typename T2>
void dia_matvec_noomp_strided(const bool overwrite_y,
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
            y_row[n * y_stride] += T2(a * diag[n]) * x_row[n * x_stride]; 
        }
    }
}



#if defined(_OPENMP)
#include "csrmv_merge.h"
#include "openmp.h"

template <typename I, typename T1, typename T2>
void dia_matvec_omp_contig(const bool overwrite_y,
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
            y_row[n] += T2(a * diag[n]) * x_row[n]; 
        }
    }
}

template <typename I, typename T1, typename T2>
void dia_matvec_omp_strided(const bool overwrite_y,
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
            y_row[n * y_stride] += T2(a * diag[n]) * x_row[n * x_stride]; 
        }
    }
}




#else



template <typename I, typename T1, typename T2>
inline void dia_matvec_omp_contig(const bool overwrite_y,
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
    dia_matvec_noomp_contig(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x,y);
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
    dia_matvec_noomp_contig(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,y_stride,y);
}


#endif


dia_matvec_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T1 a,
                        const npy_intp x_stride_bytes,
                        const T2 x[],
                        const npy_intp y_stride_bytes,
                              T2 y[])
{
    const npy_intp x_stride = x_stride_bytes/sizeof(T2);
    const npy_intp y_stride = y_stride_bytes/sizeof(T2);
    if(y_stride==1){
        if(x_stride==1){
            dia_matvec_omp_contig(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x,y);
        }
        else{
            dia_matvec_omp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,1,y);
        }
    }
    else{
        if(x_stride==1){
            dia_matvec_omp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,1,x,y_stride,y);
        }
        else{
            dia_matvec_omp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,y_stride,y);
        }
     
    }

}


#endif
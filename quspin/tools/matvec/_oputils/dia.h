#ifndef __DIA_H
#define __DIA_H 



#include <algorithm>
#include "complex_ops.h"
#include "utils.h"
#include "numpy/ndarraytypes.h"




template<typename I, typename T1, typename T2,typename T3>
void dia_matvec_noomp_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const T3 x[],
                              T3 y[])
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
        const T3 * x_row = x + j_start;
              T3 * y_row = y + i_start;

        for(I n = 0; n < N; n++){
            y_row[n] += (a * diag[n]) * x_row[n]; 
        }
    }
}

template<typename I, typename T1, typename T2,typename T3>
void dia_matvec_noomp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const npy_intp x_stride,
                        const T3 x[],
                        const npy_intp y_stride,
                              T3 y[])
{

    if(overwrite_y){
        for(I i = 0; i < n_row; i++){
            y[i * y_stride] = 0;
        }
    }

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T3 * x_row = x + j_start * x_stride;
              T3 * y_row = y + i_start * y_stride;

        for(I n = 0; n < N; n++){
            y_row[n * y_stride] += (a * diag[n]) * x_row[n * x_stride]; 
        }
    }
}


template<typename I, typename T1, typename T2,typename T3>
void dia_matvecs_noomp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
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
        for(I i = 0; i < n_diags; i++){
            const I k = offsets[i];  //diagonal offset

            const I i_start = std::max<I>(0,-k);
            const I j_start = std::max<I>(0, k);
            const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

            const I N = j_end - j_start;  //number of elements to process

            const T1 * diag = diags + (npy_intp)i*L + j_start;
            const T3 * x_row = x + j_start * x_stride_row;
                  T3 * y_row = y + i_start * y_stride_row;

            for(I n = 0; n < N; n++){
                axpy_strided((npy_intp)n_vecs,T3(a * diag[n]),x_stride_col,x_row,y_stride_col,y_row);
                x_row += x_stride_row;
                y_row += y_stride_row;
            }
        }       
    }
    else{
        for(I i = 0; i < n_diags; i++){
            const I k = offsets[i];  //diagonal offset

            const I i_start = std::max<I>(0,-k);
            const I j_start = std::max<I>(0, k);
            const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

            const I N = j_end - j_start;  //number of elements to process

            const T1 * diag = diags + (npy_intp)i*L + j_start;
            const T3 * x_start = x + j_start * x_stride_row;
                  T3 * y_start = y + i_start * y_stride_row;

            for(I m=0;m<n_vecs;m++){
                const T3 * x_row = x_start;
                      T3 * y_row = y_start;

                for(I n = 0; n < N; n++){
                    (*y_row) += (a * diag[n]) * (*x_row);
                    x_row += x_stride_row;
                    y_row += y_stride_row;
                }

                y_start += y_stride_col;
                x_start += x_stride_col;
            }
        } 
    }

}



#if defined(_OPENMP)
#include "csrmv_merge.h"
#include "openmp.h"

template<typename I, typename T1, typename T2,typename T3>
void dia_matvec_omp_contig(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
                const I offsets[], 
                const T1 diags[], 
                const T2 a,
                const T3 x[],
                      T3 y[])
{
    #pragma omp parallel
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
            const T3 * x_row = x + j_start;
                  T3 * y_row = y + i_start;

            #pragma omp for schedule(static)
            for(I n=0;n<N;n++){
                y_row[n] += (a * diag[n]) * x_row[n]; 
            }
        }
    }
}

template<typename I, typename T1, typename T2,typename T3>
void dia_matvec_omp_strided(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
                const I offsets[], 
                const T1 diags[], 
                const T2 a,
                const npy_intp x_stride,
                const T3 x[],
                const npy_intp y_stride,
                      T3 y[])
{
    #pragma omp parallel
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
            const T3 * x_row = x + j_start * x_stride;
                  T3 * y_row = y + i_start * y_stride;

            #pragma omp for schedule(static)
            for(I n=0;n<N;n++){
                y_row[n * y_stride] += (a * diag[n]) * x_row[n * x_stride]; 
            }
        }
    }
}


template<typename I, typename T1, typename T2,typename T3>
inline void dia_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
    dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}


#else



template<typename I, typename T1, typename T2,typename T3>
inline void dia_matvec_omp_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const T3 x[],
                              T3 y[])
{
    dia_matvec_noomp_contig(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x,y);
}

template<typename I, typename T1, typename T2,typename T3>
inline void dia_matvec_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const npy_intp x_stride,
                        const T3 x[],
                        const npy_intp y_stride,
                              T3 y[])
{
    dia_matvec_noomp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,y_stride,y);
}


template<typename I, typename T1, typename T2,typename T3>
inline void dia_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
    dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}

#endif

template<typename I, typename T1, typename T2,typename T3>
void dia_matvec_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const npy_intp x_stride_bytes,
                        const T3 x[],
                        const npy_intp y_stride_bytes,
                              T3 y[])
{
    const npy_intp x_stride = x_stride_bytes/sizeof(T3);
    const npy_intp y_stride = y_stride_bytes/sizeof(T3);
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

template<typename I, typename T1, typename T2,typename T3>
void dia_matvec_noomp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T2 a,
                        const npy_intp x_stride_bytes,
                        const T3 x[],
                        const npy_intp y_stride_bytes,
                              T3 y[])
{
    const npy_intp x_stride = x_stride_bytes/sizeof(T3);
    const npy_intp y_stride = y_stride_bytes/sizeof(T3);
    if(y_stride==1){
        if(x_stride==1){
            dia_matvec_noomp_contig(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x,y);
        }
        else{
            dia_matvec_noomp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,1,y);
        }
    }
    else{
        if(x_stride==1){
            dia_matvec_noomp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,1,x,y_stride,y);
        }
        else{
            dia_matvec_noomp_strided(overwrite_y,n_row,n_col,n_diags,L,offsets,diags,a,x_stride,x,y_stride,y);
        }
     
    }

}

template<typename I, typename T1, typename T2,typename T3>
inline void dia_matvecs_omp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
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
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,1,y);
        }
    }
    else if(y_stride_row==1){
        if(x_stride_col==1){
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,1,x,1,y_stride_col,y);
        }
        else if(x_stride_row==1){
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,1,x_stride_col,x,1,y_stride_col,y);
        }
        else{
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
}



template<typename I, typename T1, typename T2,typename T3>
inline void dia_matvecs_noomp(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
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
            dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            dia_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,1,y);
        }
    }
    else if(y_stride_row==1){
        if(x_stride_col==1){
            dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,1,x,1,y_stride_col,y);
        }
        else if(x_stride_row==1){
            dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,1,x_stride_col,x,1,y_stride_col,y);
        }
        else{
            dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        dia_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
}


#endif
#ifndef __MATVECS_H__
#define __MATVECS_H__

#include <complex>
#include <algorithm>
#include "numpy/ndarraytypes.h"
#include "openmp.h"
#include "utils.h"



#if defined(_OPENMP)




template <typename I, typename T1, typename T2>
void dia_matvecs_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T1 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T2 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T2 y[])
{
    if(overwrite_y){
        const npy_intp n = (npy_intp)n_row * n_vecs;

        for(npy_intp i = 0; i < n; i++){
            y[i] = 0;
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
            const T2 * x_row = x + j_start * x_stride_row;
                  T2 * y_row = y + i_start * y_stride_row;

            for(I n = 0; n < N; n++){
                axpy_strided((npy_intp)n_vecs,(T2)(a * diag[n]),x_stride_col,x_row,y_stride_col,y_row);
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
            const T2 * x_start = x + j_start * x_stride_row;
                  T2 * y_start = y + i_start * y_stride_row;

            for(I m=0;m<n_vecs;m++){
                const T2 * x_row = x_start;
                      T2 * y_row = y_start;

                for(I n = 0; n < N; n++){
                    (*y_row) += (T2)(a * diag[n]) * (*x_row);
                    x_row += x_stride_row;
                    y_row += y_stride_row;
                }

                y_start += y_stride_col;
                x_start += x_stride_col;
            }
        } 
    }

}



template<typename I, typename T1,typename T2>
void csr_matvecs_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_vecs,
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
        const npy_intp n = (npy_intp)n_row * n_vecs;

        for(npy_intp i = 0; i < n; i++){
            y[i] = 0;
        }
    }


    if(y_stride_col < y_stride_row){
        for(I k = 0; k<n_row; k++){
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                const T2 ax = a * Ax[jj];
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
                    (*y) += (T2)(a * Ax[jj]) * x[ii];
                }
                y += y_stride_row;
            }
            x += x_stride_col;
        }
    }
}




template<typename I, typename T1,typename T2>
void csc_matvecs_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_vecs,
                        const I Ap[],
                        const I Ai[],
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
        const npy_intp n = (npy_intp)n_vecs * n_row;
        for(npy_intp i = 0; i < n; i++){
            y[i] = 0;
        }
    }

    

    if(y_stride_col < y_stride_row){
        for(I j = 0; j < n_col; j++){
            I col_start = Ap[j];
            I col_end   = Ap[j+1];

            for(I ii = col_start; ii < col_end; ii++){
                T2 * y_row = y + y_stride_row * Ai[ii];
                const T2 ax = a * Ax[ii];
                axpy_strided((npy_intp)n_vecs, ax, x_stride_col, x, y_stride_col, y_row);
            }
            x += x_stride_row;
        }
    }
    else{
        for(I m=0;m<n_vecs;m++){
            const T2 * x_row = x;
            for(I j = 0; j < n_col; j++){
                I col_start = Ap[j];
                I col_end   = Ap[j+1];
                for(I ii = col_start; ii < col_end; ii++){
                    y[y_stride_row * Ai[ii]] += (T2)(a * Ax[ii]) * (*x_row);
                }
                x_row += x_stride_row;
            }
            x += x_stride_col;
            y += y_stride_col;
        }   
    }

}





#else






template <typename I, typename T1, typename T2>
void dia_matvecs_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T1 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T2 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T2 y[])
{
    if(overwrite_y){
        const npy_intp n = (npy_intp)n_row * n_vecs;

        for(npy_intp i = 0; i < n; i++){
            y[i] = 0;
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
            const T2 * x_row = x + j_start * x_stride_row;
                  T2 * y_row = y + i_start * y_stride_row;

            for(I n = 0; n < N; n++){
                axpy_strided((npy_intp)n_vecs,(T2)(a * diag[n]),x_stride_col,x_row,y_stride_col,y_row);
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
            const T2 * x_start = x + j_start * x_stride_row;
                  T2 * y_start = y + i_start * y_stride_row;

            for(I m=0;m<n_vecs;m++){
                const T2 * x_row = x_start;
                      T2 * y_row = y_start;

                for(I n = 0; n < N; n++){
                    (*y_row) += (T2)(a * diag[n]) * (*x_row);
                    x_row += x_stride_row;
                    y_row += y_stride_row;
                }

                y_start += y_stride_col;
                x_start += x_stride_col;
            }
        } 
    }

}



template<typename I, typename T1,typename T2>
void csr_matvecs_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_vecs,
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
        const npy_intp n = (npy_intp)n_row * n_vecs;

        for(npy_intp i = 0; i < n; i++){
            y[i] = 0;
        }
    }


    if(y_stride_col < y_stride_row){
        for(I k = 0; k<n_row; k++){
            for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
                const T2 ax = a * Ax[jj];
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
                    (*y) += (T2)(a * Ax[jj]) * x[ii];
                }
                y += y_stride_row;
            }
            x += x_stride_col;
        }
    }
}




template<typename I, typename T1,typename T2>
void csc_matvecs_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_vecs,
                        const I Ap[],
                        const I Ai[],
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
        const npy_intp n = (npy_intp)n_vecs * n_row;
        for(npy_intp i = 0; i < n; i++){
            y[i] = 0;
        }
    }

    

    if(y_stride_col < y_stride_row){
        for(I j = 0; j < n_col; j++){
            I col_start = Ap[j];
            I col_end   = Ap[j+1];

            for(I ii = col_start; ii < col_end; ii++){
                T2 * y_row = y + y_stride_row * Ai[ii];
                const T2 ax = a * Ax[ii];
                axpy_strided((npy_intp)n_vecs, ax, x_stride_col, x, y_stride_col, y_row);
            }
            x += x_stride_row;
        }
    }
    else{
        for(I m=0;m<n_vecs;m++){
            const T2 * x_row = x;
            for(I j = 0; j < n_col; j++){
                I col_start = Ap[j];
                I col_end   = Ap[j+1];
                for(I ii = col_start; ii < col_end; ii++){
                    y[y_stride_row * Ai[ii]] += (T2)(a * Ax[ii]) * (*x_row);
                }
                x_row += x_stride_row;
            }
            x += x_stride_col;
            y += y_stride_col;
        }      
    }

}



#endif




template <typename I, typename T1, typename T2>
void dia_matvecs(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I n_vecs,
                        const I n_diags,
                        const I L,
                        const I offsets[], 
                        const T1 diags[], 
                        const T1 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T2 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T2 y[])
{
    dia_matvecs_strided(overwrite_y,n_row,n_col,n_vecs,n_diags,L,offsets,diags,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}



template<typename I, typename T1,typename T2>
void csr_matvecs(const bool overwrite_y,
                const I n_row,
                const I n_vecs,
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
    csr_matvecs_strided(overwrite_y,n_row,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}




template<typename I, typename T1,typename T2>
void csc_matvecs(const bool overwrite_y,
                const I n_row,
                const I n_col,
                const I n_vecs,
                const I Ap[],
                const I Ai[],
                const T1 Ax[],
                const T1 a,
                const npy_intp x_stride_row,
                const npy_intp x_stride_col,
                const T2 x[],
                const npy_intp y_stride_row,
                const npy_intp y_stride_col,
                      T2 y[])
{
    csc_matvecs_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Ai,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}


#endif
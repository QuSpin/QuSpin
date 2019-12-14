#ifndef __csc_H
#define __csc_H



template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_noomp_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const T3 x[],
                              T3 y[])
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
            y[i] += (a * Ax[ii]) * x[j];
        }
    }
}

template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_noomp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride,
                        const T3 x[],
                        const npy_intp y_stride,
                              T3 y[])
{
    if(overwrite_y){
        for(I j = 0; j < n_row; j++){
            y[j * y_stride] = 0;
        }
    }

    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            const I i = Ai[ii];
            y[i * y_stride] += (a * Ax[ii]) * x[j * x_stride];
        }
    }
}


template<typename I, typename T1, typename T2,typename T3>
void csc_matvecs_noomp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Ai[], // indices for row elements in each column
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row, // X_n_row == n_col
                        const npy_intp x_stride_col, // X_n_col == n_vecs
                        const T3 x[],
                        const npy_intp y_stride_row, // Y_n_row == n_row
                        const npy_intp y_stride_col, // Y_n_col == n_vecs
                              T3 y[])
{
    if(overwrite_y){
        for(npy_intp i = 0; i < n_row; i++){
            for(npy_intp j = 0; j < n_vecs; j++){
                y[i * y_stride_row + j * y_stride_col] = 0;
            }
        }
    }

    // preference ordering of 'y' as it is being written to. 
    if(y_stride_col < y_stride_row){ 
        for(I j = 0; j < n_col; j++){
            I col_start = Ap[j];
            I col_end   = Ap[j+1];

            for(I ii = col_start; ii < col_end; ii++){
                T3 * y_row = y + y_stride_row * Ai[ii];
                const T3 ax = (a * Ax[ii]);
                axpy_strided(n_vecs, ax, x_stride_col, x, y_stride_col, y_row);
            }
            x += x_stride_row;
        }
    }
    else{
        for(I m=0;m<n_vecs;m++){
            const T3 * x_row = x;
            for(I j = 0; j < n_col; j++){
                I col_start = Ap[j];
                I col_end   = Ap[j+1];
                for(I ii = col_start; ii < col_end; ii++){
                    y[y_stride_row * Ai[ii]] += (a * Ax[ii]) * (*x_row);
                }
                x_row += x_stride_row;
            }
            x += x_stride_col;
            y += y_stride_col;
        }   
    }

}



#if defined(_OPENMP)

#include "openmp.h"



template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_omp_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const T3 x[],
                              T3 y[])
{
	#pragma omp parallel
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
	            const T3 aa = (a * Ax[ii]) * x[j];
	            atomic_add(y[i],aa);
	        }
	    }
	}
}
 

template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride,
                        const T3 x[],
                        const npy_intp y_stride,
                              T3 y[])
{
	#pragma omp parallel
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
	            const T3 aa = (a * Ax[ii]) * x[j * x_stride];
	            atomic_add(y[i * y_stride],aa);
	        }
	    }
	}
}


template<typename I, typename T1, typename T2,typename T3>
inline void csc_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
	csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Ai,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}



#else




template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_omp_contig(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const T3 x[],
                              T3 y[])
{
	csc_matvec_noomp_contig(overwrite_y,n_row,n_col,Ap,Ai,Ax,a,x,y);
}
 

template<typename I, typename T1, typename T2,typename T3>
inline void csc_matvec_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride,
                        const T3 x[],
                        const npy_intp y_stride,
                              T3 y[])
{
	csc_matvec_noomp_strided(overwrite_y,n_row,n_col,Ap,Ai,Ax,a,x_stride,x,y_stride,y);
}


template<typename I, typename T1, typename T2,typename T3>
inline void csc_matvecs_omp_strided(const bool overwrite_y,
                        const I n_row,
                        const I n_col,
                        const npy_intp n_vecs,
                        const I Ap[],
                        const I Ai[],
                        const T1 Ax[],
                        const T2 a,
                        const npy_intp x_stride_row,
                        const npy_intp x_stride_col,
                        const T3 x[],
                        const npy_intp y_stride_row,
                        const npy_intp y_stride_col,
                              T3 y[])
{
	csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Ai,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
}


#endif



 // when openmp is not being used omp and noomp versions are identical

template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_noomp(const bool overwrite_y,
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
            csc_matvec_noomp_contig(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,x,y);
        }
        else{
            csc_matvec_noomp_strided(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,x_stride,x,1,y);
        }   
    }
    else{
        if(x_stride == 1){
            csc_matvec_noomp_strided(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,1,x,y_stride,y);
        }
        else{
            csc_matvec_noomp_strided(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,x_stride,x,y_stride,y);
        }   
    }
}

template<typename I, typename T1, typename T2,typename T3>
void csc_matvec_omp(const bool overwrite_y,
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
            csc_matvec_omp_contig(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,x,y);
        }
        else{
            csc_matvec_omp_strided(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,x_stride,x,1,y);
        }   
    }
    else{
        if(x_stride == 1){
            csc_matvec_omp_strided(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,1,x,y_stride,y);
        }
        else{
            csc_matvec_omp_strided(overwrite_y,n_row,n_col,Ap,Aj,Ax,a,x_stride,x,y_stride,y);
        }   
    }
}

template<typename I, typename T1, typename T2,typename T3>
inline void csc_matvecs_noomp(const bool overwrite_y,
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
            csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,1,y);
        }
    }
    else if(y_stride_row==1){
        if(x_stride_col==1){
            csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,1,y_stride_col,y);
        }
        else if(x_stride_row==1){
            csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,1,y_stride_col,y);
        }
        else{
            csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        csc_matvecs_noomp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
    

}

template<typename I, typename T1, typename T2,typename T3>
inline void csc_matvecs_omp(const bool overwrite_y,
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
            csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,y_stride_row,1,y);
        }
        else if(x_stride_row==1){
            csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,y_stride_row,1,y);
        }
        else{
            csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,1,y);
        }
    }
    else if(y_stride_row==1){
        if(x_stride_col==1){
            csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,1,x,1,y_stride_col,y);
        }
        else if(x_stride_row==1){
            csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,1,x_stride_col,x,1,y_stride_col,y);
        }
        else{
            csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,1,y_stride_col,y);
        }
    }
    else{
        csc_matvecs_omp_strided(overwrite_y,n_row,n_col,n_vecs,Ap,Aj,Ax,a,x_stride_row,x_stride_col,x,y_stride_row,y_stride_col,y);
    }
}



#endif
#ifndef __UTILS_H
#define __UTILS_H

// y += a*x
template <typename I, typename T>
void axpy_strided(const I n, const T a,const I x_stride, const T * x,const I y_stride, T * y){
    for(I i = 0; i < n; ++i){
        (*y) += a * (*x);
        y += y_stride;
        x += x_stride;
    }
}

// y += a*x
template <typename I, typename T>
void axpy_contig(const I n, const T a,const I x_stride, const T * x,const I y_stride, T * y){
    for(I i = 0; i < n; ++i){
        (*y++) += a * (*x++);
    }
}




#endif
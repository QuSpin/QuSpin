#ifndef COMPLEX_OPS_H
#define COMPLEX_OPS_H

/*
 *  Functions to handle arithmetic operations on NumPy complex values
 */
#include <numpy/arrayobject.h>
#include <complex>

namespace basis_general
{

typedef std::complex<double> npy_cdouble_wrapper;
typedef std::complex<float> npy_cfloat_wrapper;

namespace complex_ops
{

inline npy_cdouble_wrapper exp(npy_cdouble_wrapper z){
  return std::exp(z);
}

inline npy_cfloat_wrapper exp(npy_cfloat_wrapper z){
  return std::exp(z);
}

inline npy_cdouble_wrapper conj(npy_cdouble_wrapper z){
  return std::conj(z);
}

inline npy_cfloat_wrapper conj(npy_cfloat_wrapper z){
  return std::conj(z);
} 


}


}

namespace basis_general_addition 
{



template <class c_type, class npy_type>
class complex_wrapper : public npy_type {

    public:
        /* Constructor */
        complex_wrapper( const c_type r = c_type(0), const c_type i = c_type(0) ){
            npy_type::real = r;
            npy_type::imag = i;
        }
        template<class _c_type,class _npy_type>
        complex_wrapper(const complex_wrapper<_c_type,_npy_type>& B){
            npy_type::real = B.real;
            npy_type::imag = B.imag;
        }
        /* Conversion */
        operator bool() const {
            if (npy_type::real == 0 && npy_type::imag == 0) {
                return false;
            } else {
                return true;
            }
        }
        /* Operators */
        complex_wrapper operator-() const {
          return complex_wrapper(-npy_type::real,-npy_type::imag);
        }
        complex_wrapper operator+(const complex_wrapper& B) const {
          return complex_wrapper(npy_type::real + B.real, npy_type::imag + B.imag);
        }
        complex_wrapper operator-(const complex_wrapper& B) const {
          return complex_wrapper(npy_type::real - B.real, npy_type::imag - B.imag);
        }
        complex_wrapper operator*(const complex_wrapper& B) const {
          return complex_wrapper(npy_type::real * B.real - npy_type::imag * B.imag, 
                                 npy_type::real * B.imag + npy_type::imag * B.real);
        }
        complex_wrapper operator/(const complex_wrapper& B) const {
          complex_wrapper result;
          c_type denom = 1.0 / (B.real * B.real + B.imag * B.imag);
          result.real = (npy_type::real * B.real + npy_type::imag * B.imag) * denom;
          result.imag = (npy_type::imag * B.real - npy_type::real * B.imag) * denom;
          return result;
        }
        /* in-place operators */
        complex_wrapper& operator+=(const complex_wrapper & B){
          npy_type::real += B.real;
          npy_type::imag += B.imag;
          return (*this);
        }
        complex_wrapper& operator-=(const complex_wrapper & B){
          npy_type::real -= B.real;
          npy_type::imag -= B.imag;
          return (*this);
        }
        complex_wrapper& operator*=(const complex_wrapper & B){
          c_type temp    = npy_type::real * B.real - npy_type::imag * B.imag;
          npy_type::imag = npy_type::real * B.imag + npy_type::imag * B.real;
          npy_type::real = temp;
          return (*this);
        }
        complex_wrapper& operator/=(const complex_wrapper & B){
          c_type denom   = 1.0 / (B.real * B.real + B.imag * B.imag);
          c_type temp    = (npy_type::real * B.real + npy_type::imag * B.imag) * denom; 
          npy_type::imag = (npy_type::imag * B.real - npy_type::real * B.imag) * denom;
          npy_type::real = temp;
          return (*this);
        }
        /* Boolean operations */
        bool operator==(const complex_wrapper& B) const{
          return npy_type::real == B.real && npy_type::imag == B.imag;
        }
        bool operator!=(const complex_wrapper& B) const{
          return npy_type::real != B.real || npy_type::imag != B.imag;
        }
        bool operator<(const complex_wrapper& B) const{
            if (npy_type::real == B.real){
                return npy_type::imag < B.imag;
            } else {
                return npy_type::real < B.real;
            }
        }
        bool operator>(const complex_wrapper& B) const{
            if (npy_type::real == B.real){
                return npy_type::imag > B.imag;
            } else {
                return npy_type::real > B.real;
            }
        }
        bool operator<=(const complex_wrapper& B) const{
            if (npy_type::real == B.real){
                return npy_type::imag <= B.imag;
            } else {
                return npy_type::real <= B.real;
            }
        }
        bool operator>=(const complex_wrapper& B) const{
            if (npy_type::real == B.real){
                return npy_type::imag >= B.imag;
            } else {
                return npy_type::real >= B.real;
            }
        }
        template <class T>
        bool operator==(const T& B) const{
          return npy_type::real == B && npy_type::imag == T(0);
        }
        template <class T>
        bool operator!=(const T& B) const{
          return npy_type::real != B || npy_type::imag != T(0);
        }
        template <class T>
        bool operator<(const T& B) const{
            if (npy_type::real == B) {
                return npy_type::imag < T(0);
            } else {
                return npy_type::real < B;
            }
        }
        template <class T>
        bool operator>(const T& B) const{
            if (npy_type::real == B) {
                return npy_type::imag > T(0);
            } else {
                return npy_type::real > B;
            }
        }
        template <class T>
        bool operator<=(const T& B) const{
            if (npy_type::real == B) {
                return npy_type::imag <= T(0);
            } else {
                return npy_type::real <= B;
            }
        }
        template <class T>
        bool operator>=(const T& B) const{
            if (npy_type::real == B) {
                return npy_type::imag >= T(0);
            } else {
                return npy_type::real >= B;
            }
        }
        template<class _c_type,class _npy_type>
        complex_wrapper& operator=(const complex_wrapper<_c_type,_npy_type>& B){
          npy_type::real = c_type(B.real);
          npy_type::imag = c_type(B.imag);
          return (*this);
        }
        template<class _c_type>
        complex_wrapper& operator=(const _c_type& B){
          npy_type::real = c_type(B);
          npy_type::imag = c_type(0);
          return (*this);
        }
};

typedef complex_wrapper<float,npy_cfloat> npy_cfloat_wrapper;
typedef complex_wrapper<double,npy_cdouble> npy_cdouble_wrapper;

npy_cdouble_wrapper operator*(const npy_cdouble_wrapper& A, const npy_cfloat_wrapper& B) {
  return npy_cdouble_wrapper(A.real * B.real - A.imag * B.imag, 
                             A.real * B.imag + A.imag * B.real);
}

npy_cdouble_wrapper operator*(const npy_cfloat_wrapper& A, const npy_cdouble_wrapper& B) {
  return npy_cdouble_wrapper(A.real * B.real - A.imag * B.imag, 
                             A.real * B.imag + A.imag * B.real);
}
template<typename c_type>
npy_cdouble_wrapper operator*(const npy_cdouble_wrapper& A, const c_type& B) {
  return npy_cdouble_wrapper(A.real * B, A.imag * B);
}
template<typename c_type>
npy_cdouble_wrapper operator*(const c_type& B, const npy_cdouble_wrapper& A) {
  return npy_cdouble_wrapper(A.real * B, A.imag * B);
}
npy_cdouble_wrapper operator*(const npy_cfloat_wrapper& A, const double& B) {
  return npy_cdouble_wrapper(A.real * B, A.imag * B);
}
npy_cdouble_wrapper operator*(const double& B, const npy_cfloat_wrapper& A) {
  return npy_cdouble_wrapper(A.real * B, A.imag * B);
}
npy_cfloat_wrapper operator*(const npy_cfloat_wrapper& A, const float& B) {
  return npy_cfloat_wrapper(A.real * B, A.imag * B);
}
npy_cfloat_wrapper operator*(const float& B, const npy_cfloat_wrapper& A) {
  return npy_cfloat_wrapper(A.real * B, A.imag * B);
}


template<typename c_type>
npy_cdouble_wrapper operator/(const npy_cdouble_wrapper& A, const c_type& B) {
  return npy_cdouble_wrapper(A.real / B, A.imag / B);
}

template<typename c_type>
npy_cfloat_wrapper operator/(const npy_cfloat_wrapper& A, const c_type& B) {
  return npy_cfloat_wrapper(A.real / B, A.imag / B);
}

npy_cdouble_wrapper conj(const npy_cdouble_wrapper& A) {
  return npy_cdouble_wrapper(A.real, -A.imag);
}

npy_cfloat_wrapper conj(const npy_cfloat_wrapper& A) {
  return npy_cfloat_wrapper(A.real, -A.imag);
}



inline npy_cdouble_wrapper exp(npy_cdouble_wrapper z){
  std::complex<double> res = std::exp(std::complex<double>(z.real,z.imag));
  return npy_cdouble_wrapper(res.real(),res.imag());
}


inline npy_cfloat_wrapper exp(npy_cfloat_wrapper z){
  std::complex<float> res = std::exp(std::complex<float>(z.real,z.imag));
  return npy_cfloat_wrapper(res.real(),res.imag());
}

}


#endif

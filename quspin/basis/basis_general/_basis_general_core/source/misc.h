#ifndef __MISC_H__
#define __MISC_H__

#include "numpy/ndarraytypes.h"
#include <algorithm>
#include "bits_info.h"
#include "openmp.h"


#if defined(_WIN64)

#elif defined(_WIN32)
	
#else
	#include <boost/sort/sort.hpp>
#endif


namespace basis_general {




// https://www.geeksforgeeks.org/number-swaps-sort-adjacent-swapping-allowed/
  
/* This function merges two sorted arrays and returns inversion
   count in the arrays.*/
int merge(int arr[], int temp[], const int left, const int mid, const int right)
{
    int inv_count = 0;
  
    int i = left; /* i is index for left subarray*/
    int j = mid;  /* i is index for right subarray*/
    int k = left; /* i is index for resultant merged subarray*/
    while ((i <= mid - 1) && (j <= right))
    {
        if (arr[i] >= arr[j]){
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
            /* this is tricky -- see above explanation/
              diagram for merge()*/
            inv_count = inv_count + (mid - i);
        }
    }
  
    /* Copy the remaining elements of left subarray
     (if there are any) to temp*/
    while (i <= mid - 1)
        temp[k++] = arr[i++];
  
    /* Copy the remaining elements of right subarray
     (if there are any) to temp*/
    while (j <= right)
        temp[k++] = arr[j++];
  
    /*Copy back the merged elements to original array*/
    for (i=left; i <= right; i++)
        arr[i] = temp[i];
  
    return inv_count;
}
  
/* An auxiliary recursive function that sorts the input
   array and returns the number of inversions in the
   array. */
int _mergeSort(int arr[], int temp[], const int left, const int right)
{
    int inv_count = 0;
    if (right > left)
    {
        /* Divide the array into two parts and call
          _mergeSortAndCountInv() for each of the parts */
        const int mid = (right + left)/2;
  
        /* Inversion count will be sum of inversions in
           left-part, right-part and number of inversions
           in merging */
        inv_count  = _mergeSort(arr, temp, left, mid);
        inv_count += _mergeSort(arr, temp, mid+1, right);
  
        /*Merge the two parts*/
        inv_count += merge(arr, temp, left, mid+1, right);
    }
  
    return inv_count;
}
  
/* This function sorts the input array and returns the
   number of inversions in the array */
template<class I>
int countSwaps(int arr[], const int n)
{
	int work[bit_info<I>::bits];
    return _mergeSort(arr, work, 0, n - 1);
}


template<class K,class I>
K binary_search(const K N,const I A[],const I s){
	K b,bmin,bmax;
	bmin = 0;
	bmax = N-1;
	while(bmin<=bmax){
		b = (bmax+bmin) >> 2;
		I a = A[b];
		if(s==a){
			return b;
		}
		else if(s<A[b]){
			bmin = b + 1;
		}
		else{
			bmax = b - 1;
		}
	}
	return -1;
}


template<class I>
struct compare : std::binary_function<I,I,bool>
{
	inline bool operator()(const I &a,const I &b){return a > b;}
};



template<class K,class I>
K rep_position(const npy_intp A_begin[],const npy_intp A_end[],const I A[],const npy_intp s_p,const I s){

	npy_intp begin = A_begin[s_p];
	npy_intp end = A_end[s_p];
	auto comp = compare<I>();

	if(begin<0){
		return -1;
	}
	else{
		const I * A_begin = A + begin;
		const I * A_end = A + end;
		const I * a = std::lower_bound(A_begin, A_end, s, comp);
		return ((!(a==A_end) && !comp(s,*a)) ? (K)(a-A) : -1);
	}
}

template<class K,class I>
inline K rep_position(const K N_A,const I A[],const I s){
	auto comp = compare<I>();
	const I * A_end = A+N_A;
	const I * a = std::lower_bound(A, A_end, s,comp);
	return ((!(a==A_end) && !comp(s,*a)) ? (K)(a-A) : -1);
}

bool inline check_nan(double val){
#if defined(_WIN64)
	// x64 version
	return _isnanf(val) != 0;
#elif defined(_WIN32)
	return _isnan(val) != 0;
#else
	return std::isnan(val);
#endif
}

template<class T>
inline bool equal_zero(std::complex<T> a){
	return (a.real() == 0 && a.imag() == 0);
}

template<class T>
inline bool equal_zero(T a){
	return (a == 0);
}


template<class T>
inline std::complex<double> mul(std::complex<T> a,std::complex<double> z){
	double real = a.real()*z.real() - a.imag()*z.imag();
	double imag = a.real()*z.imag() + a.imag()*z.real();

	return std::complex<double>(real,imag);
}

template<class T>
inline std::complex<double> mul(T a,std::complex<double> z){
	return std::complex<double>(a*z.real(),a*z.imag());
}




template<class T>
int inline type_checks(const std::complex<double> m){
	return 0;
}

template<>
int inline type_checks<double>(const std::complex<double> m){
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);
}

template<>
int inline type_checks<float>(const std::complex<double> m){
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);
}





template<class T>
int inline type_checks(const std::complex<double> m,std::complex<T> *M){
	M->real(m.real());
	M->imag(m.imag());
	return 0;
}

template<class T>
int inline type_checks(const std::complex<double> m,T *M){
	(*M) = m.real();
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);
}

// template<>
// int inline type_checks<signed char>(const std::complex<double> m,signed char *M){
// 	const double real = m.real();
// 	*M = (signed char) real;

// 	if(real > std::numeric_limits<signed char>::max()){
// 		return 4; // check for overflow in integer
// 	}
// 	if(std::abs(std::floor(real)-real)>1.1e-15){
// 		return 3; // check check if value is whole number
// 	}
// 	if(std::abs(m.imag())>1.1e-15){
// 		return 1; // check if imaginary part is zero
// 	}
// 	return 0;
// }

// template<>
// int inline type_checks<signed short>(const std::complex<double> m,signed short *M){
// 	const double real = m.real();
// 	*M = (signed short) real;

// 	if(real > std::numeric_limits<signed short>::max()){
// 		return 4; // check for overflow in integer
// 	}
// 	if(std::abs(std::floor(real)-real)>1.1e-15){
// 		return 3; // check check if value is whole number
// 	}
// 	if(std::abs(m.imag())>1.1e-15){
// 		return 1; // check if imaginary part is zero
// 	}
// 	return 0;
// }


template<class T>
int inline type_checks(const std::complex<double> m,const std::complex<T> v,std::complex<T> *M){
	M->real(m.real()*v.real()-m.imag()*v.imag());
	M->imag(m.imag()*v.real()+m.real()*v.imag());
	return 0;
}

template<class T>
int inline type_checks(std::complex<double> m,const T v,T *M){
	(*M) = v*m.real();
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);

}

template<class I>
struct compare_arr : std::binary_function<npy_intp,npy_intp,bool>
{
	const I * array;
	compare_arr(const I * ptr) : array(ptr) {}

	bool operator()(const npy_intp &i, const npy_intp &j) const {return array[i] > array[j];}
};

template<class I>
void argsort_decending_array(npy_intp indptr[],const I A[],const npy_intp M){

	#if defined(_WIN64)
		// x64 version
		std::sort(indptr, indptr+M, compare_arr<I>(A));
	#elif defined(_WIN32)
		std::sort(indptr, indptr+M, compare_arr<I>(A));
	#else
		#if defined(_OPENMP)
			#pragma omp parallel
			{
				const int nthread = omp_get_num_threads();
				#pragma omp master
				{
					boost::sort::block_indirect_sort(indptr, indptr+M, compare_arr<I>(A),nthread);			
				}
			}
		#else
			std::sort(indptr, indptr+M,compare_arr<I>(A));
		#endif
	#endif

}

template<class I>
bool is_decending_array(const I A[],const npy_intp M){
	// checks if array is sorted in decending order by checking if A[i] > A[i+1] for 0 <= i < M-1
	int is_sorted = 1;

	#pragma omp parallel
	{
		const int nthread = omp_get_num_threads();
		const int threadn = omp_get_thread_num();
		const npy_intp chunk = (M+nthread-2)/nthread;
		const npy_intp begin = threadn * chunk;
		const npy_intp end = std::min(chunk*(threadn+1),M-1);

		int is_sorted_thread = 1;

		for(int i=begin;i<end;i++){
			if(A[i] < A[i+1]){
				#pragma omp atomic write
				is_sorted = 0;
			}

			if(is_sorted==0){
				break;
			}

		}


	}

	return is_sorted == 1;

}






}



#endif
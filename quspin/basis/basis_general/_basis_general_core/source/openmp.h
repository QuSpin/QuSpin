#ifndef _OPENMP_H
#define _OPENMP_H

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
inline omp_int_t omp_get_max_threads() { return 1;}
#endif

#endif
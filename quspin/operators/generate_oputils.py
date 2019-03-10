
import numpy as np
from numpy import int32,int64,float32,float64,complex64,complex128
import os







numpy_ctypes={float32:"float",float64:"double",complex64:"npy_cfloat_wrapper",complex128:"npy_cdouble_wrapper",int32:"npy_int32",int64:"npy_int64"}
numpy_numtypes={float32:"NPY_FLOAT32",float64:"NPY_FLOAT64",complex64:"NPY_COMPLEX64",complex128:"NPY_COMPLEX128",int32:"NPY_INT32",int64:"NPY_INT64"}

I_types = [int32,int64]
T_types = [complex128,float64,complex64,float32]
# T_types = [complex128,float64]



get_switch_body = """
#include <iostream>

int get_switch_num(PyArray_Descr * dtype1,PyArray_Descr * dtype2,PyArray_Descr * dtype3){{
	int T1 = dtype1->type_num;
	int T2 = dtype2->type_num;
	int T3 = dtype3->type_num;
	
	{}
	return -1;
}}"""


def generate_get_switch():
	switch_num = 0
	body = "if(0){}\n"
	for T1 in I_types:
		body = body + "\telse if(PyArray_EquivTypenums(T1,{})){{\n\t\t if(0){{}}\n".format(numpy_numtypes[T1])
		for T2 in T_types:
			for T3 in T_types:
				if np.can_cast(T2,T3):
					body = body + "\t\telse if(T2=={} && T3=={}){{return {};}}\n".format(numpy_numtypes[T2],numpy_numtypes[T3],switch_num)
					switch_num += 1

		body = body + "\t\telse {return -1;}\n\t}\n"
	body = body + "\telse {return -1;}\n"

	return get_switch_body.format(body)


comp_body = """

#include "{fmt}.h"

void {fmt}_matvec_gil(const int switch_num,
					const bool overwrite_y,
					const npy_intp n_row,
					const npy_intp n_col,
						  void * Ap,
						  void * Aj,
						  void * Ax,
						  void * a,
					const npy_intp x_stride_byte,
						  void * x,
					const npy_intp y_stride_byte,
						  void * y)
{{
	switch(switch_num){{{matvec_gil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}

void {fmt}_matvec_nogil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
						    void * Ap,
						    void * Aj,
						    void * Ax,
						    void * a,
					  const npy_intp x_stride_byte,
						    void * x,
					  const npy_intp y_stride_byte,
						    void * y)
{{
	switch(switch_num){{{matvec_nogil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}

void {fmt}_matvecs_gil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
					  const npy_intp n_vecs,
						    void * Ap,
						    void * Aj,
						    void * Ax,
						    void * a,
					  const npy_intp x_stride_row_byte,
					  const npy_intp x_stride_col_byte,
						    void * x,
					  const npy_intp y_stride_row_byte,
					  const npy_intp y_stride_col_byte,
						    void * y)
{{
	switch(switch_num){{{matvecs_gil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}

void {fmt}_matvecs_nogil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
					  const npy_intp n_vecs,
						    void * Ap,
						    void * Aj,
						    void * Ax,
						    void * a,
					  const npy_intp x_stride_row_byte,
					  const npy_intp x_stride_col_byte,
						    void * x,
					  const npy_intp y_stride_row_byte,
					  const npy_intp y_stride_col_byte,
						    void * y)
{{
	switch(switch_num){{{matvecs_nogil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}"""

def generate_csr():
	switch_num = 0
	matvec_gil_body = ""
	matvec_nogil_body = ""
	matvecs_gil_body = ""
	matvecs_nogil_body = ""
	case_tmp = "\n\t\tcase {} :\n\t\t\t{}\n\t\t\tbreak;"
	matvec_tmp = "{fmt}_matvec_{omp}<{T1},{T2},{T3}>(overwrite_y,(const {T1})n_row,(const {T1})n_col,(const {T1}*)Ap,(const {T1}*)Aj,(const {T2}*)Ax,*(const {T2}*)a,x_stride_byte,(const {T3}*)x,y_stride_byte,({T3}*)y);"
	matvecs_tmp = "{fmt}_matvecs_{omp}<{T1},{T2},{T3}>(overwrite_y,(const {T1})n_row,(const {T1})n_col,n_vecs,(const {T1}*)Ap,(const {T1}*)Aj,(const {T2}*)Ax,*(const {T2}*)a,x_stride_row_byte,x_stride_col_byte,(const {T3}*)x,y_stride_row_byte,y_stride_col_byte,({T3}*)y);"
	for T1 in I_types:
		for T2 in T_types:
			for T3 in T_types:
				if np.can_cast(T2,T3):
					call = matvec_tmp.format(fmt="csr",omp="omp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvec_nogil_body = matvec_nogil_body + case_tmp.format(switch_num,call)

					call = matvec_tmp.format(fmt="csr",omp="noomp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvec_gil_body = matvec_gil_body + case_tmp.format(switch_num,call)

					call = matvecs_tmp.format(fmt="csr",omp="omp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvecs_nogil_body = matvecs_nogil_body + case_tmp.format(switch_num,call)

					call = matvecs_tmp.format(fmt="csr",omp="noomp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvecs_gil_body = matvecs_gil_body + case_tmp.format(switch_num,call)

					switch_num += 1



	return comp_body.format(fmt="csr",matvec_nogil_body=matvec_nogil_body,matvec_gil_body=matvec_gil_body,
						   matvecs_nogil_body=matvecs_nogil_body,matvecs_gil_body=matvecs_gil_body)	


def generate_csc():
	switch_num = 0
	matvec_gil_body = ""
	matvec_nogil_body = ""
	matvecs_gil_body = ""
	matvecs_nogil_body = ""
	case_tmp = "\n\t\tcase {} :\n\t\t\t{}\n\t\t\tbreak;"
	matvec_tmp = "{fmt}_matvec_{omp}<{T1},{T2},{T3}>(overwrite_y,(const {T1})n_row,(const {T1})n_col,(const {T1}*)Ap,(const {T1}*)Aj,(const {T2}*)Ax,*(const {T2}*)a,x_stride_byte,(const {T3}*)x,y_stride_byte,({T3}*)y);"
	matvecs_tmp = "{fmt}_matvecs_{omp}<{T1},{T2},{T3}>(overwrite_y,(const {T1})n_row,(const {T1})n_col,n_vecs,(const {T1}*)Ap,(const {T1}*)Aj,(const {T2}*)Ax,*(const {T2}*)a,x_stride_row_byte,x_stride_col_byte,(const {T3}*)x,y_stride_row_byte,y_stride_col_byte,({T3}*)y);"
	for T1 in I_types:
		for T2 in T_types:
			for T3 in T_types:
				if np.can_cast(T2,T3):
					call = matvec_tmp.format(fmt="csc",omp="omp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvec_nogil_body = matvec_nogil_body + case_tmp.format(switch_num,call)

					call = matvec_tmp.format(fmt="csc",omp="noomp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvec_gil_body = matvec_gil_body + case_tmp.format(switch_num,call)

					call = matvecs_tmp.format(fmt="csc",omp="omp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvecs_nogil_body = matvecs_nogil_body + case_tmp.format(switch_num,call)

					call = matvecs_tmp.format(fmt="csc",omp="noomp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvecs_gil_body = matvecs_gil_body + case_tmp.format(switch_num,call)

					switch_num += 1



	return comp_body.format(fmt="csc",matvec_nogil_body=matvec_nogil_body,matvec_gil_body=matvec_gil_body,
						   matvecs_nogil_body=matvecs_nogil_body,matvecs_gil_body=matvecs_gil_body)	


dia_body = """

#include "dia.h"

void dia_matvec_gil(const int switch_num,
					const bool overwrite_y,
					const npy_intp n_row,
					const npy_intp n_col,
                    const npy_intp n_diags,
                    const npy_intp L,
						  void * offsets,
						  void * diags,
						  void * a,
					const npy_intp x_stride_byte,
						  void * x,
					const npy_intp y_stride_byte,
						  void * y)
{{
	switch(switch_num){{{matvec_gil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}

void dia_matvec_nogil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
                      const npy_intp n_diags,
                      const npy_intp L,
						    void * offsets,
						    void * diags,
						    void * a,
					  const npy_intp x_stride_byte,
						    void * x,
					  const npy_intp y_stride_byte,
						    void * y)
{{
	switch(switch_num){{{matvec_nogil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}

void dia_matvecs_gil(const int switch_num,
					 const bool overwrite_y,
					 const npy_intp n_row,
					 const npy_intp n_col,
					 const npy_intp n_vecs,
                     const npy_intp n_diags,
                     const npy_intp L,
						   void * offsets,
						   void * diags,
						   void * a,
					 const npy_intp x_stride_row_byte,
					 const npy_intp x_stride_col_byte,
						   void * x,
					 const npy_intp y_stride_row_byte,
					 const npy_intp y_stride_col_byte,
						   void * y)
{{
	switch(switch_num){{{matvecs_gil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}

void dia_matvecs_nogil(const int switch_num,
					   const bool overwrite_y,
					   const npy_intp n_row,
					   const npy_intp n_col,
					   const npy_intp n_vecs,
                       const npy_intp n_diags,
                       const npy_intp L,
						     void * offsets,
						     void * diags,
						     void * a,
					   const npy_intp x_stride_row_byte,
					   const npy_intp x_stride_col_byte,
						     void * x,
					   const npy_intp y_stride_row_byte,
					   const npy_intp y_stride_col_byte,
						     void * y)
{{
	switch(switch_num){{{matvecs_nogil_body:}
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}}
}}"""

def generate_dia():
	switch_num = 0
	matvec_gil_body = ""
	matvec_nogil_body = ""
	matvecs_gil_body = ""
	matvecs_nogil_body = ""
	case_tmp = "\n\t\tcase {} :\n\t\t\t{}\n\t\t\tbreak;"
	matvec_tmp = "dia_matvec_{omp}(overwrite_y,(const {T1})n_row,(const {T1})n_col,(const {T1})n_diags,(const {T1})L,(const {T1}*)offsets,(const {T2}*)diags,*(const {T2}*)a,x_stride_byte,(const {T3}*)x,y_stride_byte,({T3}*)y);"
	matvecs_tmp = "dia_matvecs_{omp}(overwrite_y,(const {T1})n_row,(const {T1})n_col,n_vecs,(const {T1})n_diags,(const {T1})L,(const {T1}*)offsets,(const {T2}*)diags,*(const {T2}*)a,x_stride_row_byte,x_stride_col_byte,(const {T3}*)x,y_stride_row_byte,y_stride_col_byte,({T3}*)y);"
	for T1 in I_types:
		for T2 in T_types:
			for T3 in T_types:
				if np.can_cast(T2,T3):
					call = matvec_tmp.format(omp="omp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvec_nogil_body = matvec_nogil_body + case_tmp.format(switch_num,call)

					call = matvec_tmp.format(omp="noomp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvec_gil_body = matvec_gil_body + case_tmp.format(switch_num,call)

					call = matvecs_tmp.format(omp="omp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvecs_nogil_body = matvecs_nogil_body + case_tmp.format(switch_num,call)

					call = matvecs_tmp.format(omp="noomp",T1=numpy_ctypes[T1],T2=numpy_ctypes[T2],T3=numpy_ctypes[T3])
					matvecs_gil_body = matvecs_gil_body + case_tmp.format(switch_num,call)

					switch_num += 1



	return dia_body.format(matvec_nogil_body=matvec_nogil_body,matvec_gil_body=matvec_gil_body,
						   matvecs_nogil_body=matvecs_nogil_body,matvecs_gil_body=matvecs_gil_body)	


oputils_impl_header = """#ifndef __OPUTILS_IMPL_H__
#define __OPUTILS_IMPL_H__

#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

{header_body}
#endif"""

def generate_oputils():
	header_body = generate_get_switch()
	header_body = header_body + generate_csr()
	header_body = header_body + generate_csc()
	header_body = header_body + generate_dia()
	oputils_impl_header.format(header_body=header_body)
	path = os.path.join(os.path.dirname(__file__),"_oputils","oputils_impl.h")
	IO = open(path,"w")
	IO.write(oputils_impl_header.format(header_body=header_body))
	IO.close()


if __name__ == '__main__':
	generate_oputils()






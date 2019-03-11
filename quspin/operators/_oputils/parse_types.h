#ifndef __PARSE_TYPES_H__
#define __PARSE_TYPES_H__ 


#include "numpy/ndarrayobject.h"



int get_switch(PyArray_Descr * dtype1,PyArray_Descr * dtype2,PyArray_Descr * dtype3){
	const int t1 = dtype1->type_num;
	const int t2 = dtype2->type_num;
	const int t3 = dtype3->type_num;
	if(t1 == NPY_INT32){
		if(0){}
		else if(t2==NPY_COMPLEX128 &&  t3==NPY_COMPLEX128){return  0;}
		else if(t2==NPY_COMPLEX64  &&  t3==NPY_COMPLEX128){return  1;}
		else if(t2==NPY_COMPLEX64  &&  t3==NPY_COMPLEX64 ){return  2;}
		else if(t2==NPY_FLOAT64    &&  t3==NPY_COMPLEX128){return  3;}
		else if(t2==NPY_FLOAT64    &&  t3==NPY_FLOAT64   ){return  4;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_COMPLEX128){return  5;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_COMPLEX64 ){return  6;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_FLOAT64   ){return  7;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_FLOAT32   ){return  8;}
		else                                              {return -1;}

	}
	else if(t1 == NPY_INT64)
		if(0){}
		else if(t2==NPY_COMPLEX128 &&  t3==NPY_COMPLEX128){return   9;}
		else if(t2==NPY_COMPLEX64  &&  t3==NPY_COMPLEX128){return  10;}
		else if(t2==NPY_COMPLEX64  &&  t3==NPY_COMPLEX64 ){return  11;}
		else if(t2==NPY_FLOAT64    &&  t3==NPY_COMPLEX128){return  12;}
		else if(t2==NPY_FLOAT64    &&  t3==NPY_FLOAT64   ){return  13;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_COMPLEX128){return  14;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_COMPLEX64 ){return  15;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_FLOAT64   ){return  16;}
		else if(t2==NPY_FLOAT32    &&  t3==NPY_FLOAT32   ){return  17;}
		else                                              {return -1;}
	}
	else{return -1;}
}





#endif
import os

from quspin.tools.matvec._oputils import _matvec
from quspin.tools.matvec._oputils import _get_matvec_function


def get_matvec_function(array):
	"""Determines automatically the matrix vector product routine for an `array` based on its type.

	A highly specialized omp-parallelized application of a matrix-vector (`matvec`) product 
	depends on the array type (`csr`, `csc`, `dia`, `other` [e.g., dense]). This function determines automatically which `matvec` routine is most appropriate for a given array type. 

	Notes
	-----
	* for QuSpin builds which support OpenMP, this function will be multithreaded. 
	* see function `tools.misc.matvec()` which shows how to use the python function returned by `get_matvec_function()`. 
	* the difference between `tools.misc.matvec()` and the python function returned by `get_matvec_function` is that `tools.misc.matvec()` determines the correct matrix-vector product type every time it is called, while `get_matvec_function` allows to circumvent this extra overhead and gain some speed. 

	Examples
	--------

	The example shows how to use the `get_matvec_function()` (line 43) and the `matvec()` function (lines 47-81) in a user-defined ODE which solves the Lindblad equation for a single qubit (see also Example 17).

	.. literalinclude:: ../../doc_examples/matvec-example.py
			:linenos:
			:language: python
			:lines: 11-

	Parameters
	----------
	array : array_like object (e.g. numpy.ndarray, scipy.sparse.csr_matrix,  scipy.sparse.csc_matrix,  scipy.sparse.dia_matrix)
		Array-like object to determine the most appropriate omp-parallelized `matvec` function for. 

	Returns
	-------
	python function object
		A python function to perform the matrix-vector product. For appropriate use, see `tools.misc.matvec()`.


	"""

	return _get_matvec_function(array)




def matvec(array,other,overwrite_out=False,out=None,a=1.0):
	"""Calculates omp-parallelized matrix vector products.

	Let :math:`A` be a matrix (`array`), :math:`v` be a vector (`other`), and :math:`a` (`a`) be a scalar. This function 
	provides an omp-parallelized implementation of
	
	.. math::

		x += aAv \\qquad \\mathrm{or} \\qquad x = aAv.

	Notes
	-----
	* for QuSpin builds which support OpenMP, this function will be multithreaded. 
	* using `out=v` will result in incorrect results. 
	* `matvec` determines the correct omp-parallelized matrix-vector product, depending on the type of the input `array` (`csr`, `csc`, `dia`, `other` [e.g., dense]), every time `matvec` is called. To avoid this overhead, see `quspin.tools.misc.get_matvec_function()`.

	
	Examples
	--------

	The example shows how to use the `get_matvec_function()` (line 43) and the `matvec()` function (lines 47-81) in a user-defined ODE which solves the Lindblad equation for a single qubit (see also Example 17).

	.. literalinclude:: ../../doc_examples/matvec-example.py
			:linenos:
			:language: python
			:lines: 11-

	Parameters
	-----------
	array : array_like object (e.g. numpy.ndarray, scipy.sparse.csr_matrix,  scipy.sparse.csc_matrix,  scipy.sparse.dia_matrix)
		Sparse or dense array to take the dot product with. 
	other : array_like
		array which contains the vector to take the dot product with. 
	a : scalar, optional
		value to scale the vector with after the product with `array` is taken: :math:`x += a A v` or :math:`x = a A v`.
	out : array_like
		output array to put the results of the calculation in.
	overwrite_out : bool, optional
		If set to `True`, the function overwrites the values in `out` with the result (cf. :math:`x = a A v`). Otherwise 
		the result is added to the values in `out` (in-pace addition, cf. :math:`x += a A v`). 

	Returns
	--------
	numpy.ndarray
		result of the matrix-vector product :math:`a A v`. 

		* if `out` is not `None` and `overwrite_out = True`, the function returns `out` with the original data overwritten, otherwise if `overwrite_out = False` the result is added to `out`.
		* if `out` is `None`, the result is stored in a new array which is returned by the function. 
	

	"""


	return _matvec(array,other,overwrite_out=False,out=None,a=1.0)




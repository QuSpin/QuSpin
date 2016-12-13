

float_type={
			"type_code":"f",
			"c_matrix_type":"float",
			"np_matrix_type":"NP_FLOAT32",
			"c_complex_type":"float complex",
			"c_float_type":"float",
			"check_imag":"True"
			}

double_type={
				"type_code":"d",
				"c_matrix_type":"double",
				"np_matrix_type":"NP_FLOAT64",
				"c_complex_type":"double complex",
				"c_float_type":"double",
				"check_imag":"True"
			}

long_double_type={
					"type_code":"g",
					"c_matrix_type":"long double",
					"np_matrix_type":"NP_FLOAT128",
					"c_complex_type":"long double complex",
					"c_float_type":"long double",
					"check_imag":"True"
				}


float_complex_type={
					"type_code":"F",
					"c_matrix_type":"float complex",
					"np_matrix_type":"NP_COMPLEX64",
					"c_complex_type":"float complex",
					"c_float_type":"float",
					"check_imag":"False"
					}

double_complex_type={
					"type_code":"D",
					"c_matrix_type":"double complex",
					"np_matrix_type":"NP_COMPLEX128",
					"c_complex_type":"double complex",
					"c_float_type":"double",
					"check_imag":"False"
					}

long_double_complex_type={
						"type_code":"G",
						"c_matrix_type":"long double complex",
						"np_matrix_type":"NP_COMPLEX256",
						"c_complex_type":"long double complex",
						"c_float_type":"long double",
						"check_imag":"False"
						}



def get_templates(folder,ext):
	import os,glob
	package_dir = os.path.dirname(os.path.realpath(__file__))
	sources_dir = os.path.join(*([package_dir]+folder))
	sources=glob.glob(os.path.join(sources_dir,"*"+ext))


	return sources





def basis_ops_gen():
	import numpy
	basis_types = [
					{"np_basis_type":"NP_UINT32","c_basis_type":"unsigned int"},
				]

	matrix_types = [float_type,double_type,float_complex_type,double_complex_type]
	if hasattr(numpy,"float128"): # architecture supports long double
		matrix_types.append(long_double_type)
	if hasattr(numpy,"complex256"): # architecture supports long double complex
		matrix_types.append(long_double_complex_type)

	op_templates = get_templates(['sources','op'],".tmp")


	for op_template in op_templates:
		IO = open(op_template,"r")
		filename = op_template.replace(".tmp","")
		file_temp_str = IO.read()
		replacements = []
		for basis_type in basis_types:
			for matrix_type in matrix_types:
				replace = basis_type.copy()	
				replace.update(matrix_type)
				replacements.append(replace)


		file_str = ""
		for replace in replacements:
			file_str += file_temp_str.format(**replace)

			

		with open(filename,"w") as IO:
			IO.write(file_str)



	basis_templates = get_templates(['sources','basis'],".tmp")
	basis_templates += get_templates(['sources'],".tmp")


	for basis_template in basis_templates:
		IO = open(basis_template,"r")
		filename = basis_template.replace(".tmp","")
		file_temp_str = IO.read()

		file_str = ""
		for replace in basis_types:
			file_str += file_temp_str.format(**replace)

			

		with open(filename,"w") as IO:
			IO.write(file_str)







def cython_files():
	import os
	try:
		import Cython
		USE_CYTHON = True
	except ImportError:
		USE_CYTHON = False

	package_dir = os.path.dirname(os.path.realpath(__file__))


	if USE_CYTHON:
		basis_ops_gen()
		cython_src =  os.path.join(package_dir,"basis_ops.pyx")
		os.system("cython --cplus "+cython_src)


	return  os.path.join(package_dir,"basis_ops.cpp")
		



def configuration(parent_package='', top_path=None):
		import numpy,os
		from numpy.distutils.misc_util import Configuration
		config = Configuration('_constructors',parent_package, top_path)

		package_dir = os.path.dirname(os.path.realpath(__file__))
		src = os.path.join(package_dir,"basis_ops.cpp")
		config.add_extension('basis_ops',sources=src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")
		return config

if __name__ == '__main__':
		from numpy.distutils.core import setup
		import sys
		try:
			instr = sys.argv[1]
			if instr == "build_templates":
				cython_files()
			else:
				setup(**configuration(top_path='').todict())
		except IndexError: pass






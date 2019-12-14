#ifndef __CSR_TRACE_H__
#define __CSR_TRACE_H__ 

template <class I, class T>
T csr_trace(const I n_row,
			const I n_col, 
			const I Ap[], 
			const I Aj[], 
			const T Ax[])
{

	T trace = 0;
	const I N = (n_row<n_col?n_col:n_row);

	for(I i = 0; i < N; i++){
		const I row_start = Ap[i];
		const I row_end   = Ap[i+1];

		T diag = 0;
		for(I jj = row_start; jj < row_end; jj++){
			if (Aj[jj] == i)  // if column equals row
				diag += Ax[jj];
		}

		trace += diag;
	}
	return trace;
}


template <class I, class T>
npy_intp csr_shift_diag_pass1(const T shift,
							  const I n_row,
							  const I n_col, 
							  const I Ap[], 
							  const I Aj[], 
							  const T Ax[])
{
	const I N = (n_row<n_col?n_col:n_row);
	npy_intp nnz_new = Ap[n_row];

	for(I i = 0; i < N; i++){
		const I row_start = Ap[i];
		const I row_end   = Ap[i+1];
		bool row_has_diag = false;

		for(I jj = row_start; jj < row_end; jj++){
			if(Aj[jj]==i){
				if(Ax[jj] == -shift){
					nnz_new--;
				}
				else{
					row_has_diag=true;
				}
			}
		}

		if(!row_has_diag){ // row has no diagonal element hence shift will add one.
			nnz_new++;
		}

	}
	return nnz_new;
}


template <class I,class J, class T>
void csr_shift_diag_pass2(const T shift,
						  const I n_row,
						  const I n_col, 
						  const I Ap[], 
						  const I Aj[], 
						  const T Ax[],
								J Bp[],
								J Bj[],
								T Bx[])
{

	const I N = (n_row<n_col?n_col:n_row);
	J nnz_new = 0;
	Bp[0] = 0;
	for(I i = 0; i < N; i++){
		const I row_start = Ap[i];
		const I row_end   = Ap[i+1];
		bool diag_set = false;

		for(I jj = row_start; jj < row_end; jj++){
			if(Aj[jj]<i){
				Bj[nnz_new] = Aj[jj];
				Bx[nnz_new] = Ax[jj];
				nnz_new++;
			}
			else if(Aj[jj]==i){
				if(Ax[jj] != -shift){
					Bj[nnz_new] = Aj[jj];
					Bx[nnz_new] = Ax[jj] + shift;
					nnz_new++;
				}
				diag_set = true;
			}
			else{
				if(!diag_set){
					Bj[nnz_new] = i;
					Bx[nnz_new] = shift;
					nnz_new++;
					diag_set = true;
				}

				Bj[nnz_new] = Aj[jj];
				Bx[nnz_new] = Ax[jj];
				nnz_new++;

			}
		}
		
		if(!diag_set){
			Bj[nnz_new] = i;
			Bx[nnz_new] = shift;
			nnz_new++;			
		}

		Bp[i+1] = nnz_new;

	}
}



#include <complex>
#include <vector>
#include <algorithm>

// _np.max(_np.abs(self._A).sum(axis=0)) 
template<class I,class T>
double csr_1_norm(const I n_row,
				 const I n_col, 
				 const I Ap[], 
				 const I Aj[],
				 const std::complex<double> shift, // shift of diagonal matrix elements. 
			 	 const T Ax[])
{
	std::vector<double> col_sums(n_col);

	for(I i = 0; i < n_row; i++){
		double sum = 0;
		for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
			if(Aj[jj]==i){
				sum += std::abs(shift + std::complex<double>(Ax[jj]));
			}
			else{
				sum += std::abs(Ax[jj]);
			}
			
		}
		col_sums[i] = sum;
	}

	return *std::max_element(col_sums.begin(),col_sums.end());
}




#endif
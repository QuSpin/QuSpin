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
void csr_shift_diag(const T shift,
				    const I n_row,
				    const I n_col, 
				    const I Ap[], 
				    const I Aj[], 
				 		  T Ax[])
{

	const I N = (n_row<n_col?n_col:n_row);

	for(I i = 0; i < N; i++){
		const I row_start = Ap[i];
		const I row_end   = Ap[i+1];

		for(I jj = row_start; jj < row_end; jj++){
			if (Aj[jj] == i) // if column equals row
				Ax[jj] += shift;
		}
	}
}

#include <complex>
#include <vector>
#include <algorithm>

// _np.max(_np.abs(self._A).sum(axis=0)) 
template<class I,class T,class realT>
realT csr_1_norm(const I n_row,
				 const I n_col, 
				 const I Ap[], 
				 const I Aj[], 
			 		   T Ax[])
{
	std::vector<realT> col_sums(n_col);

	for(I i = 0; i < n_row; i++){
		for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
			col_sums[Aj[jj]] += std::abs(Ax[jj]);
		}
	}

	return *std::max_element(col_sums.begin(),col_sums.end());
}




#endif
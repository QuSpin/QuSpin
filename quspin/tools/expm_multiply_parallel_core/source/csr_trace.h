#ifndef __CSR_TRACE_H__
#define __CSR_TRACE_H__ 

template <class I, class T>
T csr_trace(const I n,
			const I n_col, 
			const I Ap[], 
			const I Aj[], 
			const T Ax[])
{

	T trace = 0;
	const I N = (n<n_col?n_col:n);

	for(I i = 0; i < N; i++){
		const I row_start = Ap[i];
		const I row_end   = Ap[i+1];

		T diag = 0;
		for(I jj = row_start; jj < row_end; jj++){
			if (Aj[jj] == i)
				diag += Ax[jj];
		}

		trace += diag;
	}
	return trace;
}


#endif
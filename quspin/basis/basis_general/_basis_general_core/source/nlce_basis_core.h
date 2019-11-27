#ifndef _NLCE_BASIS_CORE_H
#define _NLCE_BASIS_CORE_H


#include "nlce_utils.h"
#include "hcb_basis_core.h"
#include "general_basis_core.h"

#include <map>
#include <utility>
#include <iomanip>

namespace nlce {




typedef std::pair<npy_intp,UndirectedGraph> pair;

template<class I>
using map_type1 = std::map<I,npy_intp>;

template<class I>
using map_type2 = std::map<I,pair>;

template<class I>
using map_type3 = std::map<I,std::map<I,npy_intp>>;


template<class I>
class nlce_basis_core
{
private:
	basis_general::general_basis_core<I> *B_f,*B_p,*B_t;
	basis_general::general_basis_core<uint32_t> *B_pcon;
	const int * nn_list;
	const int N,Ncl,Nnn;
	std::vector<map_type1<I>> symm_clusters;
	std::vector<map_type2<I>> topo_clusters;
	std::map<I,npy_intp> index;
	map_type3<I> sub_clusters;


public:
	nlce_basis_core(const int _Ncl,const int _N,const int _Nnn,const int _nn_list[],
		const int nt_p,const int nt_t,const int maps[],const int pers[],const int qs[]) :
		nn_list(_nn_list), N(_N), Ncl(_Ncl), Nnn(_Nnn)
	{
		B_f = new basis_general::hcb_basis_core<I>(_N,nt_p+nt_t,maps,pers,qs);
		B_p = new basis_general::hcb_basis_core<I>(_N,nt_p,maps,pers,qs);
		B_t = new basis_general::hcb_basis_core<I>(_N,nt_t,maps+nt_p*_N,pers+nt_p,qs+nt_p);
		B_pcon = new basis_general::hcb_basis_core<uint32_t>(32);
		symm_clusters.resize(_Ncl);
		topo_clusters.resize(_Ncl);
	}

	~nlce_basis_core(){
		delete[] B_f;
		delete[] B_p;
		delete[] B_t;
		delete[] B_pcon;
	}

	void clusters_calc();
	// void topo_clusters_calc();
	void calc_subclusters();

	npy_intp symm_cluster_size(int);
	void symm_cluster_copy(int,I *,int *);

	npy_intp topo_cluster_size(int);
	void topo_cluster_copy(int,I *,int *);

	void get_Y_matrix_dims(npy_intp&,npy_intp&);
	void get_Y_matrix(npy_intp[],npy_intp[],npy_intp[]);

	template<class J,class K>
	void cluster_copy(J c[],K ncl[],npy_intp L[]){
		int nc = 1;
		for(auto topo_cluster : topo_clusters){
			for(auto cluster : topo_cluster){
				I s = cluster.first;
				npy_intp i = index[s];
				L[i] = cluster.second.first;
				ncl[i] = nc;

				J pos = 0;
				npy_intp j = Ncl*i;
				do {
					if(s&1){
						c[j++] = pos;
					}
					pos++;
				} while(s >>= 1);
			}
			nc++;
		}
	};
};



template<class I>
void nlce_basis_core<I>::clusters_calc(){
	int g[__GENERAL_BASIS_CORE__max_nt];
	signed char sign=0;
	UndirectedGraph graph(1);

	I s = B_f->ref_state_less((I)1,g,sign);
	symm_clusters[0][s] = 1;
	topo_clusters[0][s] = std::make_pair((int)1,graph);

	for(int i=1;i<Ncl;i++){
		build_new_symm_clusters(B_f,B_p,B_t,Nnn,nn_list,symm_clusters[i-1],symm_clusters[i]);
		build_topo_clusters<I,map_type1<I>,map_type2<I>>(Nnn,nn_list,symm_clusters[i],topo_clusters[i]);
	}

	npy_intp i = 0;
	for(auto topo_cluster : topo_clusters){
		for(auto cluster : topo_cluster){
			index[cluster.first] = i++;
		}
	}
}

template<class I>
npy_intp nlce_basis_core<I>::symm_cluster_size(int c){
	if(c >= Ncl || c < 0){
		return -1;
	}
	return symm_clusters[c].size();
}

template<class I>
void nlce_basis_core<I>::symm_cluster_copy(int c,I *c_data, int * mul_data){
	for(auto pair=symm_clusters[c].begin();pair!=symm_clusters[c].end();pair++){
		*c_data++ = pair->first;
		*mul_data++ = pair->second;
	}
}




template<class I>
npy_intp nlce_basis_core<I>::topo_cluster_size(int c){
	if(c >= Ncl || c < 0){
		return -1;
	}
	return topo_clusters[c].size();
}

template<class I>
void nlce_basis_core<I>::topo_cluster_copy(int c,I *c_data, int * mul_data){
	for(auto pair=topo_clusters[c].begin();pair!=topo_clusters[c].end();pair++){
		*c_data++ = pair->first;
		*mul_data++ = pair->second.first;
	}
}


template<class I>
void nlce_basis_core<I>::calc_subclusters()
{
	std::unordered_map<int,int> pos_to_ind;
	std::vector<int> ind_to_pos;

	for(auto topo_cluster=topo_clusters.begin();topo_cluster!=topo_clusters.end();topo_cluster++){

		const int max_cluster_size = topo_cluster - topo_clusters.begin() + 1;

		for(auto cluster : *topo_cluster){
			I s = cluster.first;
			get_ind_pos_map(s,ind_to_pos,pos_to_ind);

			for(int cluster_size=1;cluster_size<max_cluster_size;cluster_size++){
				subclusters(B_pcon,cluster_size,max_cluster_size,Nnn,nn_list,ind_to_pos,
					pos_to_ind,topo_clusters[cluster_size-1],sub_clusters[s]);
			}
		}
	}
}

template<class I>
void nlce_basis_core<I>::get_Y_matrix_dims(npy_intp &row,npy_intp &nnz){
	nnz = 0;
	row = 0;

	for(auto topo_cluster : topo_clusters){
		row += topo_cluster.size();
		for(auto cluster : topo_cluster){
			nnz += sub_clusters[cluster.first].size();
		}
	}




}

template<class I>
void nlce_basis_core<I>::get_Y_matrix(npy_intp data[],npy_intp indices[],npy_intp indptr[]){

	npy_intp nr = 0;
	npy_intp nnz = 0;
	for(auto topo_cluster : topo_clusters){
		for(auto cluster : topo_cluster){

			indptr[nr++] = nnz;
			I s = cluster.first;

			for(auto sub : sub_clusters[s]){
				indices[nnz] = index[sub.first];
				data[nnz++] = -sub.second;
			}

			indptr[nr] = nnz;
		}
	}
}





}


#endif
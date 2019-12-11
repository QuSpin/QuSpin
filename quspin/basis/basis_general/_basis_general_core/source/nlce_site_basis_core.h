#ifndef _nlce_site_basis_core_H
#define _nlce_site_basis_core_H


#include "nlce_utils.h"
#include "hcb_basis_core.h"
#include "general_basis_core.h"

#include <map>
#include <utility>
#include <iomanip>

namespace nlce_site {



template<class Graph>
using pair = std::pair<npy_intp,Graph>;

template<class I>
using map_type1 = std::map<I,npy_intp>;

template<class I,class Graph>
using map_type2 = std::map<I,pair<Graph>>;

template<class I>
using map_type3 = std::vector<map_type1<I>>;

template<class I,class Graph>
using map_type4 = std::vector<map_type2<I,Graph>>;

template<class I>
using map_type5 = std::map<I,map_type1<I>>;



template<class I>
class nlce_site_basis_core
{
private:
	basis_general::general_basis_core<I> *B_f,*B_p,*B_t;
	const int *nn_list,*nn_weight;
	const int N,Ncl,Nnn;

	map_type1<I> index;
	map_type5<I> sub_clusters;
	map_type3<I> symm_clusters;
	map_type4<I,weighted::GraphType> weighted_topo_clusters;
	map_type4<I,unweighted::GraphType> unweighted_topo_clusters;
	const bool weighted;


public:
	nlce_site_basis_core(const int _Ncl,const int _N,const int _Nnn,const int _nn_list[],
		const int nt_p,const int nt_t,const int maps[],const int pers[],const int qs[]) :
		nn_list(_nn_list), nn_weight(nullptr), N(_N), Ncl(_Ncl), Nnn(_Nnn), weighted(false)
	{
		B_f = new basis_general::hcb_basis_core<I>(_N,nt_p+nt_t,maps,pers,qs);
		B_p = new basis_general::hcb_basis_core<I>(_N,nt_p,maps,pers,qs);
		B_t = new basis_general::hcb_basis_core<I>(_N,nt_t,maps+nt_p*_N,pers+nt_p,qs+nt_p);
		symm_clusters.resize(_Ncl);
		unweighted_topo_clusters.resize(_Ncl);
	}

	nlce_site_basis_core(const int _Ncl,const int _N,const int _Nnn,const int _nn_list[],const int _nn_weight[],
		const int nt_p,const int nt_t,const int maps[],const int pers[],const int qs[]) :
		nn_list(_nn_list), nn_weight(_nn_weight), N(_N), Ncl(_Ncl), Nnn(_Nnn), weighted(true)
	{
		B_f = new basis_general::hcb_basis_core<I>(_N,nt_p+nt_t,maps,pers,qs);
		B_p = new basis_general::hcb_basis_core<I>(_N,nt_p,maps,pers,qs);
		B_t = new basis_general::hcb_basis_core<I>(_N,nt_t,maps+nt_p*_N,pers+nt_p,qs+nt_p);
		symm_clusters.resize(_Ncl);
		weighted_topo_clusters.resize(_Ncl);
	}

	~nlce_site_basis_core(){
		delete[] B_f;
		delete[] B_p;
		delete[] B_t;
	}

	void clusters_calc();
	void calc_subclusters();

	

	void get_Y_matrix_dims(npy_intp&,npy_intp&);
	void get_Y_matrix(npy_intp[],npy_intp[],npy_intp[]);

	template<class J,class K>
	void cluster_copy(J c[],K ncl[],npy_intp L[]){
		int nc = 1;
		if(weighted){
			for(auto topo_cluster : weighted_topo_clusters){
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
		}
		else{
			for(auto topo_cluster : unweighted_topo_clusters){
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
		}

	};
};



template<class I>
void nlce_site_basis_core<I>::clusters_calc(){
	int g[__GENERAL_BASIS_CORE__max_nt];
	signed char sign=0;


	I s = B_f->ref_state_less((I)1,g,sign);
	symm_clusters[0][s] = 1;
	if(weighted){
		weighted::GraphType graph(1);
		weighted_topo_clusters[0][s] = std::make_pair((int)1,graph);
	}
	else{
		unweighted::GraphType graph(1);
		unweighted_topo_clusters[0][s] = std::make_pair((int)1,graph);
	}
	

	for(int i=1;i<Ncl;i++){
		build_new_symm_clusters(B_f,B_p,B_t,Nnn,nn_list,symm_clusters[i-1],symm_clusters[i]);
		if(weighted){
			weighted::build_topo_clusters<I,map_type1<I>,map_type2<I,weighted::GraphType>>(Nnn,nn_list,nn_weight,symm_clusters[i],weighted_topo_clusters[i]);
		}
		else{
			unweighted::build_topo_clusters<I,map_type1<I>,map_type2<I,unweighted::GraphType>>(Nnn,nn_list,symm_clusters[i],unweighted_topo_clusters[i]);
		}
	}

	npy_intp i = 0;
	if(weighted){
		for(auto topo_cluster : weighted_topo_clusters){
			for(auto cluster : topo_cluster){
				index[cluster.first] = i++;
			}
		}
	}
	else{
		for(auto topo_cluster : unweighted_topo_clusters){
			for(auto cluster : topo_cluster){
				index[cluster.first] = i++;
			}
		}		
	}

}



template<class I>
void nlce_site_basis_core<I>::calc_subclusters()
{
	if(weighted){
		weighted::calc_subclusters_parallel<I,map_type4<I,weighted::GraphType>,map_type5<I>>(Nnn,nn_list,nn_weight,Ncl,weighted_topo_clusters,sub_clusters);
	}
	else{
		unweighted::calc_subclusters_parallel<I,map_type4<I,unweighted::GraphType>,map_type5<I>>(Nnn,nn_list,Ncl,unweighted_topo_clusters,sub_clusters);
	}
}




template<class I>
void nlce_site_basis_core<I>::get_Y_matrix_dims(npy_intp &row,npy_intp &nnz){
	nnz = 0;
	row = 0;

	if(weighted){
		for(auto topo_cluster : weighted_topo_clusters){
			row += topo_cluster.size();
			for(auto cluster : topo_cluster){
				nnz += sub_clusters[cluster.first].size();
			}
		}
	}
	else{
		for(auto topo_cluster : unweighted_topo_clusters){
			row += topo_cluster.size();
			for(auto cluster : topo_cluster){
				nnz += sub_clusters[cluster.first].size();
			}
		}		
	}

}

template<class I>
void nlce_site_basis_core<I>::get_Y_matrix(npy_intp data[],npy_intp indices[],npy_intp indptr[]){

	npy_intp nr = 0;
	npy_intp nnz = 0;
	if(weighted){
		for(auto topo_cluster : weighted_topo_clusters){
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
	else{
		for(auto topo_cluster : unweighted_topo_clusters){
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





}


#endif
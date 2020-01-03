#ifndef _nlce_plaquet_basis_core_H
#define _nlce_plaquet_basis_core_H


#include "nlce_utils.h"
#include "hcb_basis_core.h"
#include "general_basis_core.h"

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <iomanip>

namespace nlce {

namespace nlce_plaquet {

typedef std::pair<int,int> Edge_t;
typedef std::set<Edge_t> EdgeSet_t;
typedef std::unordered_map<int,std::map<int,EdgeSet_t>> Edges_t;
typedef std::unordered_map<int,std::map<int,int>> EdgeWeights_t;




template<class I>
class nlce_plaquet_basis_core
{
private:
	basis_general::general_basis_core<I> *B_f,*B_p,*B_t;
	EdgeWeights_t edge_weights;
	Edges_t plaquet_edges;
	const int * plaquet_sites;
 	const int N,Ncl,Nsp;

	typedef map_type1<I> IndMap_t;
	typedef map_type5<I> SubMap_t;
	typedef map_type3<I> SymmMap_t;
	typedef map_type2<I,weighted::GraphType> WTopo_t;
	typedef map_type2<I,unweighted::GraphType> UWTopo_t;
	typedef map_type4<I,weighted::GraphType> WTopoMap_t;
	typedef map_type4<I,unweighted::GraphType> UWTopoMap_t;

	IndMap_t index;
	SubMap_t sub_clusters;
	SymmMap_t symm_clusters;
	WTopoMap_t weighted_topo_clusters;
	UWTopoMap_t unweighted_topo_clusters; 
	const bool weighted;


public:
	nlce_plaquet_basis_core(const int _Ncl,const int _N,const int _plaquet_sites[],const int _Nsp,
		Edges_t &_plaquet_edges,const int nt_p,const int nt_t,const int maps[],const int pers[],
		const int qs[]) : plaquet_sites(_plaquet_sites), N(_N), Ncl(_Ncl), Nsp(_Nsp), weighted(false)
	{
		B_f = new basis_general::hcb_basis_core<I>(_N,nt_p+nt_t,maps,pers,qs);
		B_p = new basis_general::hcb_basis_core<I>(_N,nt_p,maps,pers,qs);
		B_t = new basis_general::hcb_basis_core<I>(_N,nt_t,maps+nt_p*_N,pers+nt_p,qs+nt_p);
		symm_clusters.resize(_Ncl);
		unweighted_topo_clusters.resize(_Ncl);
		plaquet_edges = _plaquet_edges;
	}

	nlce_plaquet_basis_core(const int _Ncl,const int _N,const int _plaquet_sites[],const int _Nsp,
		Edges_t &_plaquet_edges,EdgeWeights_t &_edge_weights,const int nt_p,const int nt_t,
		const int maps[],const int pers[],const int qs[]) : 
		plaquet_sites(_plaquet_sites), N(_N), Ncl(_Ncl), Nsp(_Nsp), weighted(false)
	{
		B_f = new basis_general::hcb_basis_core<I>(_N,nt_p+nt_t,maps,pers,qs);
		B_p = new basis_general::hcb_basis_core<I>(_N,nt_p,maps,pers,qs);
		B_t = new basis_general::hcb_basis_core<I>(_N,nt_t,maps+nt_p*_N,pers+nt_p,qs+nt_p);
		symm_clusters.resize(_Ncl);
		unweighted_topo_clusters.resize(_Ncl);
		plaquet_edges = _plaquet_edges;
		edge_weights = _edge_weights;
	}

	~nlce_plaquet_basis_core(){
		delete[] B_f;
		delete[] B_p;
		delete[] B_t;
	}

	void clusters_calc();
	void calc_subclusters();

	

	void get_Y_matrix_dims(npy_intp&,npy_intp&);
	void get_Y_matrix(npy_intp[],npy_intp[],npy_intp[]);

	template<class J,class K>
	void cluster_copy(npy_intp c_row,J c[],K ncl[],npy_intp L[]){
		int nc = 0;
		if(weighted){
			for(auto topo_cluster : weighted_topo_clusters){
				for(auto cluster : topo_cluster){
					I s = cluster.first;
					npy_intp i = index[s];
					L[i] = cluster.second.first;
					ncl[i] = nc;

					J pos = 0;
					npy_intp j = c_row*i;
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
					npy_intp j = c_row*i;
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
void nlce_plaquet_basis_core<I>::clusters_calc(){
	int g[__GENERAL_BASIS_CORE__max_nt];
	signed char sign=0;


	I s = B_f->ref_state_less((I)0,g,sign);
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
		build_new_symm_clusters(B_f,B_p,B_t,plaquet_edges,symm_clusters[i-1],symm_clusters[i]);
		if(weighted){
			weighted::build_topo_clusters<I,Edges_t,EdgeWeights_t,IndMap_t,WTopo_t>(Nsp,plaquet_sites,plaquet_edges,edge_weights,symm_clusters[i],weighted_topo_clusters[i]);
		}
		else{
			unweighted::build_topo_clusters<I,Edges_t,IndMap_t,UWTopo_t>(Nsp,plaquet_sites,plaquet_edges,symm_clusters[i],unweighted_topo_clusters[i]);
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
void nlce_plaquet_basis_core<I>::calc_subclusters()
{
	if(weighted){
		weighted::calc_subclusters_parallel<I,Edges_t,EdgeWeights_t,WTopoMap_t,SubMap_t>(Nsp,plaquet_sites,plaquet_edges,edge_weights,Ncl,weighted_topo_clusters,sub_clusters);
	}
	else{
		unweighted::calc_subclusters_parallel<I,Edges_t,UWTopoMap_t,SubMap_t>(Nsp,plaquet_sites,plaquet_edges,Ncl,unweighted_topo_clusters,sub_clusters);
	}
}




template<class I>
void nlce_plaquet_basis_core<I>::get_Y_matrix_dims(npy_intp &row,npy_intp &nnz){
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
void nlce_plaquet_basis_core<I>::get_Y_matrix(npy_intp data[],npy_intp indices[],npy_intp indptr[]){

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

}

#endif
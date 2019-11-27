#ifndef _NLCE_UTILS_H
#define _NLCE_UTILS_H



#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/isomorphism.hpp>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <utility>
#include <stack>

#include "general_basis_core.h"
#include "openmp.h"

namespace nlce {




template<class I,class Set>
void mult_core(basis_general::general_basis_core<I> * B_p,
			   basis_general::general_basis_core<I> * B_t,
			   Set &clusters,
			   int g[],
			   signed char sign,
			   I s,
			   const int depth=1)
{
	const int nt = B_p->get_nt();
	if(nt==0){
		return;
	}

	const int symm = depth - 1;
	const int per = B_p->pers[symm];


	if(depth < nt){
		for(int i=0;i<per;i++){
			mult_core(B_p,B_t,clusters,g,sign,s,depth+1);
			s = B_p->map_state(s,symm,sign);			
		}
	}
	else{
		for(int i=0;i<per;i++){
			s = B_t->ref_state_less(s,g,sign);
			clusters.insert(s); // insert new integer into list
			s = B_p->map_state(s,symm,sign);
		}
	}
}

template<class I>
int mult(basis_general::general_basis_core<I> * B_p,
		 basis_general::general_basis_core<I> * B_t,
		 int g[],
		 signed char sign,
		 I s)
{
	std::unordered_set<I> clusters = {};

	mult_core(B_p,B_t,clusters,g,sign,s);
	return clusters.size();
}


template<class I,class map_type>
void build_new_symm_clusters(basis_general::general_basis_core<I> * B_f,
							 basis_general::general_basis_core<I> * B_p,
			 				 basis_general::general_basis_core<I> * B_t,
			 				 const int Nnn,
			 				 const int nn_list[],
			 				 const map_type &seed_clusters,
			 				 	   map_type &new_clusters)

{
	const int nthread = omp_get_max_threads();
	std::vector<map_type> new_clusters_thread_vec(nthread);
	map_type * new_clusters_thread = &new_clusters_thread_vec[0];

	#pragma omp parallel 
	{
		int g[__GENERAL_BASIS_CORE__max_nt];
		signed char sign=0;
		const int threadn = omp_get_thread_num();

		npy_intp cnt = 0;


		for(auto pair=seed_clusters.begin();pair!=seed_clusters.end();pair++,cnt++)
		{
			if(cnt%nthread != threadn) continue;

			const I seed = pair->first;
			const int * nn_begin = nn_list;
			const int * nn_end   = nn_list + Nnn;

			I s = seed;

			do {
				if(s&1){
					for(const int * nn_p = nn_begin; nn_p != nn_end;nn_p++){
						int nn = *nn_p;

						if(nn < 0)
							continue;

						bool occ = (seed>>nn)&1;
						
						if(!occ){
							I r = seed ^ ((I)1 << nn);

							r = B_f->ref_state_less(r,g,sign);

							auto pos = new_clusters_thread[threadn].find(r);
							if(pos == new_clusters_thread[threadn].end()){
								int mul = mult(B_p,B_t,g,sign,r);
								new_clusters_thread[threadn][r] = mul;
							}
						}
					}
				}

				nn_begin += Nnn;
				nn_end += Nnn;

			} while(s >>= 1);

		}
	}

	for(auto list : new_clusters_thread_vec){
		new_clusters.insert(list.begin(),list.end());
	}


}

typedef boost::adjacency_list<boost::setS,boost::vecS,boost::undirectedS> UndirectedGraph;
typedef boost::graph_traits<UndirectedGraph>::vertex_descriptor VertexDescr;


template<class I,class Stack,class VertexList,class Graph>
void build_adj(Stack &stack,
			   VertexList &verts,
			   Graph &graph,
			   const I s,
			   const int Nnn,
			   const int nn_list[])
{
	graph.clear();
	verts.clear();

	while(!stack.empty()){
		stack.pop();
	}

	if(s==0){
		return;
	}

	I r = s;

	int site = 0;

	do {
		if(r&1){
			verts[site] = boost::add_vertex(graph);
		}
		site++;
	} while(r >>= 1);

	stack.push(verts.begin()->first);

	r = s;

	while(!verts.empty()){

		site = stack.top(); stack.pop();
		
		int shift = site * Nnn;
		const int * nn_begin = nn_list  + shift;
		const int * nn_end   = nn_begin + Nnn;

		for(const int * nn_p = nn_begin; nn_p != nn_end; nn_p++){
			int nn = *nn_p;

			if(nn < 0)
				continue;

			auto iter = verts.find(nn);
			if(iter != verts.end()){
				boost::add_edge(verts[site],verts[nn],graph);
				stack.push(nn);
			}
		}

		verts.erase(site);
	}
}

template<class I,class map_type1,class map_type2>
void build_topo_clusters(const int Nnn,
		 				 const int nn_list[],
		 				 const map_type1 &symm_clusters,
		 					   map_type2 &topo_clusters)
{
	std::unordered_map<int,VertexDescr> verts;
	UndirectedGraph graph;
	std::stack<int> stack;

	for(auto pair : symm_clusters)
	{	
		I s = pair.first;

		build_adj(stack,verts,graph,s,Nnn,nn_list);

		bool found = false;
		for(auto cls : topo_clusters)
		{
			found = boost::isomorphism(cls.second.second,graph);

			if(found){
				topo_clusters[cls.first].first += pair.second;
				break;
			}
		}

		if(!found){
			topo_clusters[s] = std::make_pair(pair.second,graph);
		}


	}
}


template<class I,class Vec,class Map>
void get_ind_pos_map(const I s,
					 Vec &ind_to_pos,
					 Map &pos_to_ind)
{
	I r = s;
	pos_to_ind.clear();
	ind_to_pos.clear();

	int pos = 0;
	int ind = 0;

	do {
		if(r&1){
			pos_to_ind[pos] = ind++;
			ind_to_pos.push_back(pos);
		}
		pos++;
	} while(r >>= 1);
}

template<class I,class Vec,class Map,
	class Stack,class VertexList,class Graph>
bool is_connected(const I c,
				  const int Nnn,
				  const int nn_list[],
				  const Vec &ind_to_pos,
				  const Map &pos_to_ind,
				  Stack &stack,
				  VertexList &verts,
				  Graph &graph)
{
	verts.clear();
	graph.clear();

	while(!stack.empty()){
		stack.pop();
	}

	// notation:
	// ind: position of bit in the integer, not correspinding to the index in space.
	// pos: position of bit in space as mapped into the cluster we are checking. 

	// start at some initially occupied point
	// remove sites from a cluster starting from
	// that position until no neighboring sites
	// are occupied. If cluster is connected,
	// then the result should have no sites and
	// the integer should be 0.

	I r = c;

	int ind = 0;

	do {
		if(r&1){
			verts[ind] = boost::add_vertex(graph);
		}
		ind++;
	} while(r >>= 1);


	// add that ind to stack
	stack.push(verts.begin()->first);
	// reset integer.
	r = c;

	while(!stack.empty()){
		int ind = stack.top(); stack.pop();
		int pos = ind_to_pos[ind];

		int shift = pos * Nnn;
		const int * nn_begin = nn_list  + shift;
		const int * nn_end   = nn_begin + Nnn;

		for(const int * nn_p=nn_begin;nn_p!=nn_end;nn_p++){
			int nn_pos = *nn_p;
			if(nn_pos < 0)
				continue;

			// check if pos maps to cluster
			auto iter2 = pos_to_ind.find(nn_pos);
			if(iter2 != pos_to_ind.end()){ 

				// get ind for given pos
				int nn_ind = iter2->second; 
				
				if((r>>nn_ind)&1){ 
				// if position is occupied 
				// add to stack and add edge to graph

					stack.push(nn_ind);
					boost::add_edge(verts[ind],verts[nn_ind],graph);
				}
			}
		}

		r ^= ((I)1 << ind); // remove site from cluster

	}

	return (r==0);
}




template<class J,class Vec,class Map,
			class map_type1,class map_type2>
void subclusters(basis_general::general_basis_core<J> * B,
				 const int ClusterSize,
				 const int MaxClusterSize,
				 const int Nnn,
				 const int nn_list[],
				 const Vec &ind_to_pos,
				 const Map &pos_to_ind,
				 const map_type1 &topo_clusters,
					   map_type2 &sub_clusters)
{
	std::unordered_map<int,VertexDescr> verts;
	UndirectedGraph graph;
	std::stack<int> stack;
	int g[__GENERAL_BASIS_CORE__max_nt];
	signed char sign=0;


	J c_max = 0;
	J c = 0;
	for(int i=0;i<ClusterSize;i++){
		c_max += ((J)1 << (MaxClusterSize - i - 1));
		c += ((J)1 << i);
	}

	while(c <= c_max){
		
		bool connected = is_connected(c,Nnn,nn_list,ind_to_pos,pos_to_ind,stack,verts,graph);

		if(connected){
			// check to see if this subcluster has been encountered, using graph.
			bool found = false;
			for(auto cls : topo_clusters){

				found = boost::isomorphism(cls.second.second,graph);
				if(found){
					sub_clusters[cls.first] += 1;
					break;
				}
			}

			// if(!found){
			// 	std::cout << "not found!" << std::endl;
			// }
		}
		c = B->next_state_pcon(c,0);
	}
	// std::cout << std::endl;
}





}

#endif
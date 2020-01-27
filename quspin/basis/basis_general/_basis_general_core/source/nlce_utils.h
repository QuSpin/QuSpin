#ifndef _NLCE_UTILS_H
#define _NLCE_UTILS_H



#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/isomorphism.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <utility>
#include <stack>

#include "general_basis_core.h"
#include "openmp.h"



namespace nlce {

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

class vf2_counter
{

	int * count;

public:
	
	vf2_counter(int * _count) : count(_count) {}

	template <typename CorrespondenceMap1To2,
			  typename CorrespondenceMap2To1>
	bool operator()(CorrespondenceMap1To2 f, CorrespondenceMap2To1 g) {
		(*count)++;
		return true;
	}
};




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

template<class I,class Vec>
void get_ind_pos_map(const I s,
					 Vec &ind_to_pos)
{
	I r = s;
	ind_to_pos.clear();

	int pos = 0;

	do {
		if(r&1){
			ind_to_pos.push_back(pos);
		}
		pos++;
	} while((r >>= 1) > 0);
}


namespace nlce_site {


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
				if((s&1)==1){
					for(const int * nn_p = nn_begin; nn_p != nn_end;nn_p++){
						int nn = *nn_p;

						if(nn < 0)
							continue;

						bool occ = ((seed>>nn)&1) == 1;
						
						if(!occ){
							I r = seed ^ ((I)1 << nn);

							r = B_f->ref_state_less(r,g,sign);

							if(new_clusters_thread[threadn].count(r)==0){
								int mul = mult(B_p,B_t,g,sign,r);
								new_clusters_thread[threadn][r] = mul;
							}
						}
					}
				}

				nn_begin += Nnn;
				nn_end += Nnn;

			} while((s >>= 1) > 0);

		}
	}

	for(auto list : new_clusters_thread_vec){
		new_clusters.insert(list.begin(),list.end());
	}
}



namespace weighted {

typedef boost::property<boost::edge_weight_t, int> Weight;
typedef boost::adjacency_list<boost::setS,boost::vecS,boost::undirectedS,
						boost::no_property,Weight> GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor VertexDescr;



template<class I,class Stack,class VertexList,class Graph,class Set>
void build_adj(Stack &stack,
			   VertexList &verts,
			   Graph &graph,
			   Set &visited,
			   const I s,
			   const int Nnn,
			   const int nn_list[],
			   const int nn_weight[])
{
	graph.clear();
	verts.clear();
	visited.clear();

	while(!stack.empty()){
		stack.pop();
	}

	I r = s;

	int pos = 0;

	do {
		if(r&1){
			verts[pos] = boost::add_vertex(graph);
		}
		pos++;
	} while((r >>= 1) > 0);

	stack.push(verts.begin()->first);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const int * nn = nn_list  + pos * Nnn;
		visited.insert(pos);

		for(int i=0;i<Nnn;i++){
			const int nn_pos = nn[i];

			if(nn_pos < 0)
				continue;

			if(visited.count(nn_pos)==0 && verts.count(nn_pos)!=0){
				boost::add_edge(verts[pos],verts[nn_pos],Weight(nn_weight[i]),graph);
				stack.push(nn_pos);					
			}
		}
	}
}

template<class I,class map_type1,class map_type2>
void build_topo_clusters(const int Nnn,
		 				 const int nn_list[],
		 				 const int nn_weight[],
		 				 const map_type1 &symm_clusters,
		 					   map_type2 &topo_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> topo_clusters_thread_vec(nthread);
	map_type2 * topo_clusters_thread = &topo_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::unordered_map<int,VertexDescr> verts;
		GraphType graph;
		std::stack<int> stack;
		std::unordered_set<int> visited;
		npy_intp cnt = 0;
	
		for(auto pair=symm_clusters.begin();pair!=symm_clusters.end();pair++,cnt++)
		{	
			if(cnt%nthread != threadn) continue;

			const I s = pair->first;
	
			build_adj(stack,verts,graph,visited,s,Nnn,nn_list,nn_weight);
	
			bool found = false;
	
			for(auto iter=topo_clusters_thread[threadn].begin();iter!=topo_clusters_thread[threadn].end();iter++)
			{
				found = boost::isomorphism(iter->second.second,graph);
	
				if(found){
					topo_clusters_thread[threadn][iter->first].first += pair->second;
					break;
				}
			}
	
			if(!found){
				topo_clusters_thread[threadn][s] = std::make_pair(pair->second,graph);
			}
		}
	}

	for(auto list=topo_clusters_thread_vec.begin();list!=topo_clusters_thread_vec.end();list++){

		for(auto iter1=list->begin();iter1!=list->end();iter1++){
			bool found = false;
			for(auto iter2=topo_clusters.begin();iter2!=topo_clusters.end();iter2++)
			{
				found = boost::isomorphism(iter2->second.second,iter1->second.second);
	
				if(found){
					topo_clusters[iter2->first].first += iter1->second.first;;
					break;
				}
			}
	
			if(!found){
				topo_clusters[iter1->first] = iter1->second;
			}
		}
	}
}




template<class I,class Vec,
	class Stack,class VertexList,class Graph,class Set>
bool is_connected(Stack &stack,
				  VertexList &verts,
				  Graph &graph,
				  Set &visited,
				  const I c,
				  const int Nnn,
				  const int nn_list[],
				  const int nn_weight[],
				  const Vec &ind_to_pos)
{
	verts.clear();
	graph.clear();
	visited.clear();

	while(!stack.empty()){
		stack.pop();
	}

	// notation:
	// ind: position of bit in the integer, not correspinding to the index in space.
	// pos: position of bit in space as mapped into the cluster we are checking. 

	I r = c;

	int ind = 0;

	do {
		if(r&1){
			verts[ind_to_pos[ind]] = boost::add_vertex(graph);
		}
		ind++;
	} while((r >>= 1) > 0);


	// add that ind to stack
	stack.push(verts.begin()->first);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const int * nn = nn_list  + pos * Nnn;
		visited.insert(pos);

		for(int i=0;i<Nnn;i++){
			const int nn_pos = nn[i];

			if(nn_pos < 0)
				continue;

			if(visited.count(nn_pos)==0 && verts.count(nn_pos)!=0){
				boost::add_edge(verts[pos],verts[nn_pos],Weight(nn_weight[i]),graph);
				stack.push(nn_pos);					
			}
		}
	}

	return visited.size() == verts.size();
}



template<class Vec,class map_type1,class map_type2>
void subclusters(const int ClusterSize,
				 const int MaxClusterSize,
				 const int Nnn,
				 const int nn_list[],
				 const int nn_weight[],
				 const Vec &ind_to_pos,
				 const map_type1 &topo_clusters,
					   map_type2 &sub_clusters)
{
	std::unordered_map<int,VertexDescr> verts;
	GraphType graph;
	std::stack<int> stack;
	std::unordered_set<int> visited;

	uint_fast32_t c_max = (uint_fast32_t)1 << MaxClusterSize;
	uint_fast32_t c = 0, t = 0;
	for(int i=0;i<ClusterSize;i++){
		c += ((uint_fast32_t)1 << i);
	}

	while(c < c_max){
		bool connected = is_connected(stack,verts,graph,visited,c,Nnn,nn_list,nn_weight,ind_to_pos);
		if(connected){
			// check to see if this subcluster has been encountered, using graph.
			bool found = false;

			for(auto iter=topo_clusters.begin();iter!=topo_clusters.end();iter++){
				found = boost::isomorphism(iter->second.second,graph);
				if(found){
					sub_clusters[iter->first] += 1;
					break;
				}
			}
		}
		t = (c | (c - 1)) + 1;
		c = t | ((((t & (0-t)) / (c & (0-c))) >> 1) - 1);
	}
}


template<class I,class map_type1,class map_type2>
void calc_subclusters_parallel(const int Nnn,
							   const int nn_list[],
							   const int nn_weight[],
							   const int Ncl,
							   const map_type1 &topo_clusters,
									 map_type2 &sub_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> sub_clusters_thread_vec(nthread);
	map_type2 * sub_clusters_thread = &sub_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::vector<int> ind_to_pos;

		ind_to_pos.reserve(Ncl+1);

		int mcs = 1;		
		for(int j=0;j<Ncl;j++){
			auto begin = topo_clusters[j].begin();
			auto end   = topo_clusters[j].end();
			npy_intp cnt = 0;

			for(auto pair=begin;pair!=end;pair++,cnt++)
			{
				if(cnt%nthread != threadn) continue;

				const I s = pair->first;
				get_ind_pos_map(s,ind_to_pos);
				
				for(int cs=1;cs<mcs;cs++){
					subclusters(cs,mcs,Nnn,nn_list,nn_weight,ind_to_pos,
						topo_clusters[cs-1],sub_clusters_thread[threadn][s]);
				}

			}
			mcs++;
		}
	}

	for(auto list : sub_clusters_thread_vec){
		sub_clusters.insert(list.begin(),list.end());
	}
}




}





namespace unweighted {

typedef boost::adjacency_list<boost::setS,boost::vecS,boost::undirectedS> GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor VertexDescr;




template<class I,class Stack,class VertexList,class Graph,class Set>
void build_adj(Stack &stack,
			   VertexList &verts,
			   Graph &graph,
			   Set &visited,
			   const I s,
			   const int Nnn,
			   const int nn_list[])
{
	graph.clear();
	verts.clear();
	visited.clear();

	while(!stack.empty()){
		stack.pop();
	}

	I r = s;

	int pos = 0;

	do {
		if(r&1){
			verts[pos] = boost::add_vertex(graph);
		}
		pos++;
	} while((r >>= 1) > 0);

	stack.push(verts.begin()->first);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const int * nn_begin = nn_list  + pos * Nnn;
		const int * nn_end   = nn_begin + Nnn;
		visited.insert(pos);

		for(const int * nn_p = nn_begin; nn_p != nn_end; nn_p++){
			const int nn_pos = *nn_p;

			if(nn_pos < 0)
				continue;

			if(visited.count(nn_pos)==0 && verts.count(nn_pos)!=0){
				boost::add_edge(verts[pos],verts[nn_pos],graph);
				stack.push(nn_pos);					
			}
		}
	}
}

template<class I,class map_type1,class map_type2>
void build_topo_clusters(const int Nnn,
		 				 const int nn_list[],
		 				 const map_type1 &symm_clusters,
		 					   map_type2 &topo_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> topo_clusters_thread_vec(nthread);
	map_type2 * topo_clusters_thread = &topo_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::unordered_map<int,VertexDescr> verts;
		GraphType graph;
		std::stack<int> stack;
		std::unordered_set<int> visited;
		npy_intp cnt = 0;
	
		for(auto pair=symm_clusters.begin();pair!=symm_clusters.end();pair++,cnt++)
		{	
			if(cnt%nthread != threadn) continue;

			const I s = pair->first;
	
			build_adj(stack,verts,graph,visited,s,Nnn,nn_list);
	
			bool found = false;
	
			for(auto iter=topo_clusters_thread[threadn].begin();iter!=topo_clusters_thread[threadn].end();iter++)
			{
				found = boost::isomorphism(iter->second.second,graph);
	
				if(found){
					topo_clusters_thread[threadn][iter->first].first += pair->second;
					break;
				}
			}
	
			if(!found){
				topo_clusters_thread[threadn][s] = std::make_pair(pair->second,graph);
			}
		}
	}

	for(auto list=topo_clusters_thread_vec.begin();list!=topo_clusters_thread_vec.end();list++){

		for(auto iter1=list->begin();iter1!=list->end();iter1++){
			bool found = false;
			for(auto iter2=topo_clusters.begin();iter2!=topo_clusters.end();iter2++)
			{
				found = boost::isomorphism(iter2->second.second,iter1->second.second);
	
				if(found){
					topo_clusters[iter2->first].first += iter1->second.first;;
					break;
				}
			}
	
			if(!found){
				topo_clusters[iter1->first] = iter1->second;
			}
		}
	}


}



template<class I,class Vec,
	class Stack,class VertexList,class Graph,class Set>
bool is_connected(Stack &stack,
				  VertexList &verts,
				  Graph &graph,
				  Set &visited,
				  const I c,
				  const int Nnn,
				  const int nn_list[],
				  const Vec &ind_to_pos)
{
	verts.clear();
	graph.clear();
	visited.clear();

	while(!stack.empty()){
		stack.pop();
	}

	// notation:
	// ind: position of bit in the integer, not correspinding to the index in space.
	// pos: position of bit in space as mapped into the cluster we are checking. 

	I r = c;

	int ind = 0;

	do {
		if(r&1){
			verts[ind_to_pos[ind]] = boost::add_vertex(graph);
		}
		ind++;
	} while((r >>= 1) > 0);


	// add that ind to stack
	stack.push(verts.begin()->first);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const int * nn_begin = nn_list  + pos * Nnn;
		const int * nn_end   = nn_begin + Nnn;
		visited.insert(pos);

		for(const int * nn_p = nn_begin; nn_p != nn_end; nn_p++){
			const int nn_pos = *nn_p;

			if(nn_pos < 0)
				continue;

			if(visited.count(nn_pos)==0 && verts.count(nn_pos)!=0){
				boost::add_edge(verts[pos],verts[nn_pos],graph);
				stack.push(nn_pos);					
			}
		}
	}

	return visited.size() == verts.size();
}




template<class Vec,class map_type1,class map_type2>
void subclusters(const int ClusterSize,
				 const int MaxClusterSize,
				 const int Nnn,
				 const int nn_list[],
				 const Vec &ind_to_pos,
				 const map_type1 &topo_clusters,
					   map_type2 &sub_clusters)
{
	std::unordered_map<int,VertexDescr> verts;
	GraphType graph;
	std::stack<int> stack;
	std::unordered_set<int> visited;

	uint_fast32_t c_max = (uint_fast32_t)1 << MaxClusterSize;
	uint_fast32_t c = 0, t = 0;
	for(int i=0;i<ClusterSize;i++){
		c += ((uint_fast32_t)1 << i);
	}

	while(c < c_max){
		bool connected = is_connected(stack,verts,graph,visited,c,Nnn,nn_list,ind_to_pos);
		if(connected){
			// check to see if this subcluster has been encountered, using graph.
			bool found = false;

			for(auto iter=topo_clusters.begin();iter!=topo_clusters.end();iter++){
				found = boost::isomorphism(iter->second.second,graph);
				if(found){
					sub_clusters[iter->first] += 1;
					break;
				}
			}
		}
		t = (c | (c - 1)) + 1;
		c = t | ((((t & (0-t)) / (c & (0-c))) >> 1) - 1);
	}
}

template<class I,class map_type1,class map_type2>
void calc_subclusters_parallel(const int Nnn,
							   const int nn_list[],
							   const int Ncl,
							   const map_type1 &topo_clusters,
									 map_type2 &sub_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> sub_clusters_thread_vec(nthread);
	map_type2 * sub_clusters_thread = &sub_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::vector<int> ind_to_pos;

		ind_to_pos.reserve(Ncl+1);

		int mcs = 1;		
		for(int j=0;j<Ncl;j++){
			auto begin = topo_clusters[j].begin();
			auto end   = topo_clusters[j].end();
			npy_intp cnt = 0;

			for(auto pair=begin;pair!=end;pair++,cnt++)
			{
				if(cnt%nthread != threadn) continue;

				const I s = pair->first;
				get_ind_pos_map(s,ind_to_pos);
				
				for(int cs=1;cs<mcs;cs++){
					subclusters(cs,mcs,Nnn,nn_list,ind_to_pos,
						topo_clusters[cs-1],sub_clusters_thread[threadn][s]);
				}

			}
			mcs++;
		}
	}

	for(auto list : sub_clusters_thread_vec){
		sub_clusters.insert(list.begin(),list.end());
	}
}


}

}


namespace nlce_plaquet {

template<class I,class map_type,class Edges_t>
void build_new_symm_clusters(basis_general::general_basis_core<I> * B_f,
							 basis_general::general_basis_core<I> * B_p,
			 				 basis_general::general_basis_core<I> * B_t,
			 				 const Edges_t &plaquet_edges,
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

			I s = seed;
			if(s==0){
				new_clusters_thread[threadn][(I)1] = 1;
			}
			else{
				int pos = 0;
				do {
					if((s&1)==1){
						const auto * edge_dict = &plaquet_edges.at(pos);
						for(auto it = edge_dict->begin();it!=edge_dict->end();it++){
							int nn = it->first;
							if(nn != pos){
								bool occ = ((seed>>nn)&1) == 1;
								
								if(!occ){
									I r = seed ^ ((I)1 << nn);

									r = B_f->ref_state_less(r,g,sign);

									if(new_clusters_thread[threadn].count(r)==0){
										int mul = mult(B_p,B_t,g,sign,r);
										new_clusters_thread[threadn][r] = mul;
									}
								}
							}
						}
					}

					pos++;

				} while((s >>= 1) > 0);
			}

		}
	}

	for(auto list : new_clusters_thread_vec){
		new_clusters.insert(list.begin(),list.end());
	}
}



namespace weighted {

typedef boost::property<boost::edge_weight_t, int> Weight;
typedef boost::adjacency_list<boost::setS,boost::vecS,boost::undirectedS,
						boost::no_property,Weight> GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor VertexDescr;



template<class I,class Stack,class VertexList,class Graph,
		class Set,class Edges_t,class EdgeWeights_t>
void build_adj(Stack &stack,
			   VertexList &verts,
			   Graph &graph,
			   Set &visited,
			   Set &plaquet_list,
			   const Edges_t &plaquet_edges,
			   const EdgeWeights_t &edge_weights,
			   const I s,
			   const int Nsp,
			   const int plaquet_sites[])
{
	graph.clear();
	verts.clear();
	visited.clear();
	plaquet_list.clear();

	while(!stack.empty()){
		stack.pop();
	}

	I r = s;

	int pos = 0;
	int plaquet_pos = 0;

	do {
		if(r&1){
			const int * p_sites = plaquet_sites + pos * Nsp;
			for(int i=0;i<Nsp;i++){
				int site = p_sites[i];
				if(verts.count(site)==0){
					verts[site] = boost::add_vertex(graph);	
				}
			}
			plaquet_list.insert(pos);
			plaquet_pos = pos;
		}
		pos++;
	} while((r >>= 1) > 0);

	stack.push(plaquet_pos);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const auto * edge_dict = &plaquet_edges.at(pos);
		for(auto it1=edge_dict->begin();it1!=edge_dict->end();it1++){
			int new_pos = it1->first;

			if(visited.count(new_pos)==0 && plaquet_list.count(new_pos)!=0){
				const auto * edge_set = &it1->second;
				
				for(auto it2=edge_set->begin();it2!=edge_set->end();it2++){
					const int i = it2->first;
					const int j = it2->second;
					const int w = (edge_weights.at(i)).at(j);
					boost::add_edge(verts[i],verts[j],Weight(w),graph);
				}
				
				stack.push(new_pos);
			}		
		}

		visited.insert(pos);
	}
}




template<class I,class Edges_t,class EdgeWeights_t,class map_type1,class map_type2>
void build_topo_clusters(const int Nsp,
		 				 const int plaquet_sites[],
		 				 const Edges_t &plaquet_edges,
		 				 const EdgeWeights_t &edge_weights,
		 				 const map_type1 &symm_clusters,
		 					   map_type2 &topo_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> topo_clusters_thread_vec(nthread);
	map_type2 * topo_clusters_thread = &topo_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::unordered_map<int,VertexDescr> verts;
		GraphType graph;
		std::stack<int> stack;
		std::unordered_set<int> visited,plaquet_list;
		npy_intp cnt = 0;
	
		for(auto pair=symm_clusters.begin();pair!=symm_clusters.end();pair++,cnt++)
		{	
			if(cnt%nthread != threadn) continue;

			const I s = pair->first;
	
			build_adj(stack,verts,graph,visited,plaquet_list,plaquet_edges,edge_weights,s,Nsp,plaquet_sites);
	
			bool found = false;
	
			for(auto iter=topo_clusters_thread[threadn].begin();iter!=topo_clusters_thread[threadn].end();iter++)
			{
				found = boost::isomorphism(iter->second.second,graph);
	
				if(found){
					topo_clusters_thread[threadn][iter->first].first += pair->second;
					break;
				}
			}
	
			if(!found){
				topo_clusters_thread[threadn][s] = std::make_pair(pair->second,graph);
			}
		}
	}

	for(auto list=topo_clusters_thread_vec.begin();list!=topo_clusters_thread_vec.end();list++){

		for(auto iter1=list->begin();iter1!=list->end();iter1++){
			bool found = false;
			for(auto iter2=topo_clusters.begin();iter2!=topo_clusters.end();iter2++)
			{
				found = boost::isomorphism(iter2->second.second,iter1->second.second);
	
				if(found){
					topo_clusters[iter2->first].first += iter1->second.first;;
					break;
				}
			}
	
			if(!found){
				topo_clusters[iter1->first] = iter1->second;
			}
		}
	}
}


template<class I,class Vec,class Stack,class VertexList,
		class Graph,class Set,class Edges_t,class EdgeWeights_t>
bool is_connected(Stack &stack,
				  VertexList &verts,
				  Graph &graph,
				  Set &visited,
				  Set &plaquet_list,
				  const Edges_t &plaquet_edges,
				  const EdgeWeights_t &edge_weights,
				  const I s,
				  const int Nsp,
				  const int plaquet_sites[],
				  const Vec &ind_to_pos)
{
	graph.clear();
	verts.clear();
	visited.clear();
	plaquet_list.clear();

	while(!stack.empty()){
		stack.pop();
	}

	I r = s;

	int ind = 0;
	int plaquet_pos = 0;

	do {
		if(r&1){
			int pos = ind_to_pos[ind];
			const int * p_sites = plaquet_sites + pos * Nsp;
			for(int i=0;i<Nsp;i++){
				int site = p_sites[i];
				if(verts.count(site)==0){
					verts[site] = boost::add_vertex(graph);	
				}
			}
			plaquet_list.insert(pos);
			plaquet_pos = pos;
		}
		ind++;
	} while((r >>= 1) > 0);

	stack.push(plaquet_pos);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const auto * edge_dict = &plaquet_edges.at(pos);
		for(auto it1=edge_dict->begin();it1!=edge_dict->end();it1++){
			int new_pos = it1->first;

			if(visited.count(new_pos)==0 && plaquet_list.count(new_pos)!=0){
				const auto * edge_set = &it1->second;
				
				for(auto it2=edge_set->begin();it2!=edge_set->end();it2++){
					const int i = it2->first;
					const int j = it2->second;
					const int w = (edge_weights.at(i)).at(j);
					boost::add_edge(verts[i],verts[j],Weight(w),graph);
				}
				
				stack.push(new_pos);
			}		
		}

		visited.insert(pos);
	}

	return visited.size() == plaquet_list.size();
}



template<class I,class Edges_t,class EdgeWeights_t,class Vec,class map_type1,class map_type2>
void subclusters(const int ClusterSize,
				 const int MaxClusterSize,
				 const int Nsp,
				 const int plaquet_sites[],
				 const I s,
		 		 const Edges_t &plaquet_edges,
		 		 const EdgeWeights_t &edge_weights,
				 const Vec &ind_to_pos,
				 const map_type1 &topo_clusters,
					   map_type2 &sub_clusters)
{
	std::unordered_map<int,VertexDescr> verts;
	GraphType graph;
	std::stack<int> stack;
	std::unordered_set<int> visited,plaquet_list;

	if(ClusterSize==0){ // single site subcluster
		build_adj(stack,verts,graph,visited,plaquet_list,plaquet_edges,edge_weights,s,Nsp,plaquet_sites);
		sub_clusters[0] = boost::num_vertices(graph);
		return;
	}

	uint_fast32_t c_max = (uint_fast32_t)1 << MaxClusterSize;
	uint_fast32_t c = 0, t = 0;
	for(int i=0;i<ClusterSize;i++){
		c += ((uint_fast32_t)1 << i);
	}

	while(c < c_max){
		bool connected = is_connected(stack,verts,graph,visited,plaquet_list,plaquet_edges,edge_weights,c,Nsp,plaquet_sites,ind_to_pos);
		if(connected){
			// check to see if this subcluster has been encountered, using graph.
			bool found = false;

			for(auto iter=topo_clusters.begin();iter!=topo_clusters.end();iter++){
				found = boost::isomorphism(iter->second.second,graph);
				if(found){
					sub_clusters[iter->first] += 1;
					break;
				}
			}
		}
		t = (c | (c - 1)) + 1;
		c = t | ((((t & (0-t)) / (c & (0-c))) >> 1) - 1);
	}
}

template<class I,class Edges_t,class EdgeWeights_t,class map_type1,class map_type2>
void calc_subclusters_parallel(const int Nsp,
							   const int plaquet_sites[],
		 					   const Edges_t &plaquet_edges,
		 					   const EdgeWeights_t &edge_weights,
							   const int Ncl,
							   const map_type1 &topo_clusters,
									 map_type2 &sub_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> sub_clusters_thread_vec(nthread);
	map_type2 * sub_clusters_thread = &sub_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::vector<int> ind_to_pos;

		ind_to_pos.reserve(Ncl+1);

		int mcs = 0;		
		for(int j=0;j<Ncl;j++){
			auto begin = topo_clusters[j].begin();
			auto end   = topo_clusters[j].end();
			npy_intp cnt = 0;

			for(auto pair=begin;pair!=end;pair++,cnt++)
			{
				if(cnt%nthread != threadn) continue;

				const I s = pair->first;
				get_ind_pos_map(s,ind_to_pos);
				
				for(int cs=0;cs<mcs;cs++){
					subclusters(cs,mcs,Nsp,plaquet_sites,s,plaquet_edges,edge_weights,ind_to_pos,
						topo_clusters[cs],sub_clusters_thread[threadn][s]);
				}

			}
			mcs++;
		}
	}

	for(auto list : sub_clusters_thread_vec){
		sub_clusters.insert(list.begin(),list.end());
	}
}




}





namespace unweighted {

typedef boost::adjacency_list<boost::setS,boost::vecS,boost::undirectedS> GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor VertexDescr;



template<class I,class Stack,class VertexList,class Graph,class Set,class Edges_t>
void build_adj(Stack &stack,
			   VertexList &verts,
			   Graph &graph,
			   Set &visited,
			   Set &plaquet_list,
			   const Edges_t &plaquet_edges,
			   const I s,
			   const int Nsp,
			   const int plaquet_sites[])
{
	graph.clear();
	verts.clear();
	visited.clear();
	plaquet_list.clear();

	while(!stack.empty()){
		stack.pop();
	}

	I r = s;

	int pos = 0;

	do {
		if(r&1){
			const int * p_sites = plaquet_sites + pos * Nsp;
			for(int i=0;i<Nsp;i++){
				int site = p_sites[i];
				if(verts.count(site)==0){
					verts[site] = boost::add_vertex(graph);	
				}
			}
			plaquet_list.insert(pos);
		}
		pos++;
	} while((r >>= 1) > 0);

	stack.push(*plaquet_list.begin());

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const auto * edge_dict = &plaquet_edges.at(pos);
		for(auto it1=edge_dict->begin();it1!=edge_dict->end();it1++){
			int new_pos = it1->first;

			if(visited.count(new_pos)==0 && plaquet_list.count(new_pos)!=0){
				const auto * edge_set = &it1->second;
				
				for(auto it2=edge_set->begin();it2!=edge_set->end();it2++){
					const int i = it2->first;
					const int j = it2->second;
					boost::add_edge(verts[i],verts[j],graph);
				}
				
				stack.push(new_pos);
			}		
		}

		visited.insert(pos);
	}
}


template<class I,class Edges_t,class map_type1,class map_type2>
void build_topo_clusters(const int Nsp,
		 				 const int plaquet_sites[],
		 				 const Edges_t &plaquet_edges,
		 				 const map_type1 &symm_clusters,
		 					   map_type2 &topo_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> topo_clusters_thread_vec(nthread);
	map_type2 * topo_clusters_thread = &topo_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::unordered_map<int,VertexDescr> verts;
		GraphType graph;
		std::stack<int> stack;
		std::unordered_set<int> visited,plaquet_list;
		npy_intp cnt = 0;
	
		for(auto pair=symm_clusters.begin();pair!=symm_clusters.end();pair++,cnt++)
		{	
			if(cnt%nthread != threadn) continue;

			const I s = pair->first;
	
			build_adj(stack,verts,graph,visited,plaquet_list,plaquet_edges,s,Nsp,plaquet_sites);
	
			bool found = false;
	
			for(auto iter=topo_clusters_thread[threadn].begin();iter!=topo_clusters_thread[threadn].end();iter++)
			{
				found = boost::isomorphism(iter->second.second,graph);
	
				if(found){
					topo_clusters_thread[threadn][iter->first].first += pair->second;
					break;
				}
			}
	
			if(!found){
				topo_clusters_thread[threadn][s] = std::make_pair(pair->second,graph);
			}
		}
	}

	for(auto list=topo_clusters_thread_vec.begin();list!=topo_clusters_thread_vec.end();list++){

		for(auto iter1=list->begin();iter1!=list->end();iter1++){
			bool found = false;
			for(auto iter2=topo_clusters.begin();iter2!=topo_clusters.end();iter2++)
			{
				found = boost::isomorphism(iter2->second.second,iter1->second.second);
	
				if(found){
					topo_clusters[iter2->first].first += iter1->second.first;;
					break;
				}
			}
	
			if(!found){
				topo_clusters[iter1->first] = iter1->second;
			}
		}
	}
}


template<class I,class Vec,
	class Stack,class VertexList,class Graph,class Set,class Edges_t>
bool is_connected(Stack &stack,
				  VertexList &verts,
				  Graph &graph,
				  Set &visited,
				  Set &plaquet_list,
				  const Edges_t &plaquet_edges,
				  const I s,
				  const int Nsp,
				  const int plaquet_sites[],
				  const Vec &ind_to_pos)
{
	graph.clear();
	verts.clear();
	visited.clear();
	plaquet_list.clear();

	while(!stack.empty()){
		stack.pop();
	}

	I r = s;

	int ind = 0;
	int plaquet_pos = 0;

	do {
		if(r&1){
			int pos = ind_to_pos[ind];
			const int * p_sites = plaquet_sites + pos * Nsp;
			for(int i=0;i<Nsp;i++){
				int site = p_sites[i];
				if(verts.count(site)==0){
					verts[site] = boost::add_vertex(graph);	
				}
			}
			plaquet_list.insert(pos);
			plaquet_pos = pos;
		}
		ind++;
	} while((r >>= 1) > 0);

	stack.push(plaquet_pos);

	while(!stack.empty()){
		const int pos = stack.top(); stack.pop();
		const auto * edge_dict = &plaquet_edges.at(pos);
		for(auto it1=edge_dict->begin();it1!=edge_dict->end();it1++){
			int new_pos = it1->first;

			if(visited.count(new_pos)==0 && plaquet_list.count(new_pos)!=0){
				const auto * edge_set = &it1->second;
				
				for(auto it2=edge_set->begin();it2!=edge_set->end();it2++){
					const int i = it2->first;
					const int j = it2->second;
					boost::add_edge(verts[i],verts[j],graph);
				}
				
				stack.push(new_pos);
			}		
		}

		visited.insert(pos);
	}

	return visited.size() == plaquet_list.size();
}



template<class I,class Edges_t,class Vec,class map_type1,class map_type2>
void subclusters(const int ClusterSize,
				 const int MaxClusterSize,
				 const int Nsp,
				 const int plaquet_sites[],
				 const I s,
		 		 const Edges_t &plaquet_edges,
				 const Vec &ind_to_pos,
				 const map_type1 &topo_clusters,
					   map_type2 &sub_clusters)
{
	std::unordered_map<int,VertexDescr> verts;
	GraphType graph;
	std::stack<int> stack;
	std::unordered_set<int> visited,plaquet_list;

	if(ClusterSize==0){ // single site subcluster
		build_adj(stack,verts,graph,visited,plaquet_list,plaquet_edges,s,Nsp,plaquet_sites);
		sub_clusters[0] = boost::num_vertices(graph);
		return;
	}

	uint_fast32_t c_max = (uint_fast32_t)1 << MaxClusterSize;
	uint_fast32_t c = 0, t = 0;
	for(int i=0;i<ClusterSize;i++){
		c += ((uint_fast32_t)1 << i);
	}

	while(c < c_max){
		bool connected = is_connected(stack,verts,graph,visited,plaquet_list,plaquet_edges,c,Nsp,plaquet_sites,ind_to_pos);
		if(connected){
			// check to see if this subcluster has been encountered, using graph.
			bool found = false;

			for(auto iter=topo_clusters.begin();iter!=topo_clusters.end();iter++){
				found = boost::isomorphism(iter->second.second,graph);
				if(found){
					sub_clusters[iter->first] += 1;
					break;
				}
			}
		}
		t = (c | (c - 1)) + 1;
		c = t | ((((t & (0-t)) / (c & (0-c))) >> 1) - 1);
	}
}


template<class I,class Edges_t,class map_type1,class map_type2>
void calc_subclusters_parallel(const int Nsp,
							   const int plaquet_sites[],
		 					   const Edges_t &plaquet_edges,
							   const int Ncl,
							   const map_type1 &topo_clusters,
									 map_type2 &sub_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> sub_clusters_thread_vec(nthread);
	map_type2 * sub_clusters_thread = &sub_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::vector<int> ind_to_pos;

		ind_to_pos.reserve(Ncl+1);

		int mcs = 0;		
		for(int j=0;j<Ncl;j++){
			auto begin = topo_clusters[j].begin();
			auto end   = topo_clusters[j].end();
			npy_intp cnt = 0;

			for(auto pair=begin;pair!=end;pair++,cnt++)
			{
				if(cnt%nthread != threadn) continue;

				const I s = pair->first;
				get_ind_pos_map(s,ind_to_pos);
				
				for(int cs=0;cs<mcs;cs++){
					subclusters(cs,mcs,Nsp,plaquet_sites,s,plaquet_edges,ind_to_pos,
						topo_clusters[cs],sub_clusters_thread[threadn][s]);
				}

			}
			mcs++;
		}
	}

	for(auto list : sub_clusters_thread_vec){
		sub_clusters.insert(list.begin(),list.end());
	}
}


/*

template<class Graph,class map_type1,class map_type2>
void subclusters(const int Npg,
				 const Graph graph,
				 const map_type1 &topo_clusters,
					   map_type2 &sub_clusters)
{

	auto begin = topo_clusters.begin();
	auto end   = topo_clusters.end();

	for(auto pair=begin;pair!=end;pair++)
	{
		int count=0;

		bool found_subgraph = boost::vf2_subgraph_iso(pair->second.second,graph,vf2_counter(&count));
		
		const int subgraph_size = boost::num_vertices(pair->second.second);

		if(found_subgraph){
			if(boost::num_vertices(pair->second.second) > 1){
				sub_clusters[pair->first] = count/Npg;
			}
			else{
				sub_clusters[pair->first] = count;
			}
			
		}
	}
}

template<class I,class Edges_t,class map_type1,class map_type2>
void calc_subclusters_parallel(const int Npg,
							   const int Nsp,
							   const int plaquet_sites[],
		 					   const Edges_t &plaquet_edges,
							   const int Ncl,
							   const map_type1 &topo_clusters,
									 map_type2 &sub_clusters)
{
	const int nthread = omp_get_max_threads();

	std::vector<map_type2> sub_clusters_thread_vec(nthread);
	map_type2 * sub_clusters_thread = &sub_clusters_thread_vec[0];

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		std::vector<int> ind_to_pos;


		int mcs = 0;		
		for(int j=0;j<Ncl;j++){
			auto begin = topo_clusters[j].begin();
			auto end   = topo_clusters[j].end();
			npy_intp cnt = 0;



			for(auto pair=begin;pair!=end;pair++,cnt++)
			{
				if(cnt%nthread != threadn) continue;
				const I s = pair->first;

				for(int cs=0;cs<mcs;cs++){
					subclusters(Npg,pair->second.second,topo_clusters[cs],sub_clusters_thread[threadn][s]);
				}

			}
			mcs++;
		}
	}

	for(auto list : sub_clusters_thread_vec){
		sub_clusters.insert(list.begin(),list.end());
	}
}

*/

}

}

}








#endif
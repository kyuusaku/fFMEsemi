#include "mtc.h"

struct Edge_ *get_random_successor( struct Edge_ *x, float degree )
{
	float s = degree*rand()/RAND_MAX;
	struct Edge_ *p = x;
	while( p->w < s ){
		s -= p->w;
		if(!p->next) return p;
		p = p->next;
	}
	return p;
}

//reference: D.B. Wilson. Generating random spanning trees more quickly than the cover time.
void random_spanning_tree(struct Edge *graph, int *nodes, int n_nodes, int n_edges, struct Edge *tree, int total)
{
	struct Edge_ *pool = (struct Edge_ *)malloc(2*n_edges*sizeof(struct Edge_));
	struct Edge_ **g = (struct Edge_ **)malloc(total*sizeof(struct Edge_ *));
	build_adj_graph(graph, total, n_edges, g, pool);

	int i, idx;
	float *degree = (float *)malloc(total*sizeof(float));
	memset(degree, 0, total*sizeof(int));	
	for( i=0; i<n_nodes; i++ ){
		idx = nodes[i];
		struct Edge_ *ptr = g[idx];
		while( ptr ){
			degree[idx] += ptr->w;
			ptr = ptr->next;
		}
	}

	int r = floor(((double)rand())/((double)RAND_MAX)*n_nodes);	
	struct Edge_ **next = (struct Edge_ **)malloc(total*sizeof(struct Edge_ *));	
	bool *is_in_tree = (bool *)malloc(total*sizeof(bool));
	for( i=0; i<total; i++ ) is_in_tree[i] = false;
	is_in_tree[nodes[r]] = true;
	
	int cnt=0;
	for( i=0; i<n_nodes; i++ ){
		idx = nodes[i];
		while( !is_in_tree[idx] ){
			next[idx] = get_random_successor(g[idx], degree[idx]);
			idx = next[idx]->idx;
		}
		idx = nodes[i];
		while( !is_in_tree[idx] ){
			is_in_tree[idx] = true;
			tree[cnt].x = idx;
			tree[cnt].y = next[idx]->idx;
			tree[cnt].w = next[idx]->w;
			cnt ++;

			idx = next[idx]->idx;
		}
	}

	free(is_in_tree);
	free(next);
	free(pool);
	free(g);
	free(degree);

}
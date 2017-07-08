#include "mtc.h"

int stack_sz;
int *stack;
void DFS(struct Edge_ **g, int idx, bool *visited, struct Edge *ec, int *e_cnt_, int *nc, int *n_cnt_)
{
	int e_cnt=0, n_cnt=0;

	while(stack_sz){
		int idx = stack[stack_sz-1];
		stack_sz--;
		if( visited[idx] == true ) continue;

		visited[idx] = true;
		nc[n_cnt++] = idx;
		for(struct Edge_ *ptr=g[idx]; ptr; ptr=ptr->next){
			if(visited[ptr->idx]==false){
				ec[e_cnt].x = ptr->idx;
				ec[e_cnt].y = idx;
				ec[e_cnt++].w = ptr->w;

				stack[stack_sz++] = ptr->idx;
			}		
		}
	}
	
	*n_cnt_ = n_cnt;
	*e_cnt_ = e_cnt;
}


//return the number of components
int detect_disconnected_components(struct Edge *graph, int n_nodes, int n_edges, struct Edge *e_com, int *e_com_sz, int *n_com, int *n_com_sz)
{
	struct Edge_ *pool = (struct Edge_ *)malloc(2*n_edges*sizeof(struct Edge_));
	struct Edge_ **g = (struct Edge_ **)malloc(n_nodes*sizeof(struct Edge_ *));
	build_adj_graph(graph, n_nodes, n_edges, g, pool);

	bool *visited = (bool *)malloc(n_nodes*sizeof(bool));
	memset(visited, 0, n_nodes);

	stack = (int *)malloc(n_edges*sizeof(int));
	stack_sz = 0;

	int comp_cnt=0, node_cnt=0, edge_cnt=0;
	for(int i=0; i<n_nodes; i++){
		if(!visited[i]){
			stack[stack_sz++] = i;
			int e, n;
			DFS(g, i, visited, e_com+edge_cnt, &e, n_com+node_cnt, &n);
			e_com_sz[comp_cnt] = e;
			n_com_sz[comp_cnt++] = n;

			edge_cnt += e;
			node_cnt += n;
		}
	}
	if(edge_cnt!=n_edges||n_nodes!=node_cnt) printf("error in detecting disconnected component.\n");
	free(visited);
	free(stack);
	free(pool);
	free(g);
	return comp_cnt;
}



#include "mtc.h"

void build_adj_graph(struct Edge *graph, int n_nodes, int n_edges, struct Edge_ **g, struct Edge_ *pool)
{
	memset(g, 0, n_nodes*sizeof(struct Edge_ *));
	for(int i=0; i<n_edges; i++){
		int j = 2*i;
		pool[j].idx = graph[i].x;
		pool[j].w = graph[i].w;
		pool[j].next = g[graph[i].y];
		g[graph[i].y] = pool + j;
		j++;
		pool[j].idx = graph[i].y;
		pool[j].w = graph[i].w;
		pool[j].next = g[graph[i].x];
		g[graph[i].x] = pool + j;
	}

}

int *heap, *position, size;
float *dist;

void heap_filterup(int p)
{
	int idx = heap[p];
	float key = dist[idx];
	int i=(p-1)/2;

	while( p>0 ){
		if(dist[heap[i]]<=key)
			break;
		else{
			heap[p] = heap[i];
			position[heap[p]] = p;
			p = i;
			i = (p-1)/2;
		}
	}
	heap[p] = idx;
	position[idx] = p;
}

void heap_filterdown(int p)
{
	int idx = heap[p];
	float key = dist[idx];
	int i = 2*p+1;

	while(i<=size-1){
		if( i<size-1 && dist[heap[i]]>dist[heap[i+1]] ) i++;
		if( key<=dist[heap[i]] )
			break;
		else{
			heap[p] = heap[i];
			position[heap[p]] = p;
			p = i;
			i = 2*p + 1;
		}
	}
	heap[p] = idx;
	position[idx] = p;
}

void shortest_path_tree( struct Edge *graph, int *nodes, int n_nodes, int n_edges, struct Edge *tree, int total, int s )
{
	struct Edge_ *pool = (struct Edge_ *)malloc(2*n_edges*sizeof(struct Edge_));
	struct Edge_ **g = (struct Edge_ **)malloc(total*sizeof(struct Edge_ *));
	build_adj_graph(graph, total, n_edges, g, pool);

	//initialize the heap
	int i, j;
	dist = (float *)malloc(total*sizeof(float));//distances to sourse
	heap = (int *)malloc(n_nodes*sizeof(int));
	int *prev =(int *)malloc(2*total*sizeof(int));
	position = prev + total;

	for(i=0; i<total; i++) dist[i] = INFI;
	struct Edge_ *ptr = g[s];
	while(ptr){
		dist[ptr->idx] = ptr->w;
		prev[ptr->idx] = s;
		ptr = ptr->next;
	}
	for( i=0, j=0; i<n_nodes-1; i++ ){
		if( nodes[i]==s ) j++;
		heap[i] = nodes[j++];
		heap_filterup(i);				
	}
	size = n_nodes-1;

	//find shorest path
	int cnt = 0;
	while(size>0){
		i = heap[0];
		j = prev[i];		
		struct Edge_ *adj = g[i];
		while(j!=adj->idx) adj = adj->next;
		tree[cnt].x = i;
		tree[cnt].y = j;
		tree[cnt++].w = adj->w;

		heap[0] = heap[size-1];
		position[heap[0]] = 0;
		size--;
		heap_filterdown(0);
		adj = g[i];
		while(adj){
			j = adj->idx;
			if( j != s && dist[j] > dist[i]+adj->w ){
				dist[j] = dist[i] + adj->w;
				prev[j] = i;
				heap_filterup(position[j]);
			}
			adj = adj->next;
		}		
	}

	free(prev);
	free(dist);
	free(heap);
	free(pool);
	free(g);
}
/*
void main()
{
	struct Edge graph[9] = {{0,1,4},{0,2,2},{1,2,1},{1,3,2},{1,4,3},{2,3,4},{2,4,5},{4,3,1}};
	struct Edge *tree_graph;
	shortest_path_tree(graph, 5, 8, 0, &tree_graph);

	return;
}*/
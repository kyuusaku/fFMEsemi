#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mtc.h"

extern int n_lb, n_pure;
extern struct Node *lb_trees[MAX_ARRAY_LEN];
extern struct Node *pure_trees[MAX_ARRAY_LEN];

//input: a tree stored in the adjacent matrix
//output: a tree stored in linked nodes
void build_tree(struct Edge_ **tree, int n_node, int r, struct Node **root, int total)
{
	bool *visited = (bool *)malloc(total*sizeof(bool));
	for(int i=0; i<total; i++) visited[i] = false;

	*root = (struct Node *)malloc(sizeof(struct Node));
	(*root)->idx = r;
	(*root)->first_chi = NULL;
	(*root)->next_bro = NULL;
	(*root)->w = -1;
	visited[r] = true;

	struct Node **stack = (struct Node **)malloc(n_node*sizeof(struct Node *));
	stack[0] = *root;
	int stack_sz = 1;

	while( stack_sz ){
	
		struct Node *ptr = stack[--stack_sz];
		int idx = ptr->idx;

		struct Edge_ *cur = tree[idx];
		if(visited[cur->idx]) cur = cur->next;
		if( cur ){
			visited[cur->idx] = true;
			ptr->first_chi = (struct Node *)malloc(sizeof(struct Node));
			ptr = ptr->first_chi;
			ptr->first_chi = NULL;
			ptr->next_bro = NULL;
			ptr->idx = cur->idx;
			ptr->w = cur->w;
			
			stack[stack_sz++] = ptr;
			cur = cur->next;
		}

		while( cur ){
			if(!visited[cur->idx]){
				visited[cur->idx] = true;
				ptr->next_bro = (struct Node *)malloc(sizeof(struct Node));
				ptr = ptr->next_bro;
				ptr->first_chi = NULL;
				ptr->next_bro = NULL;
				ptr->idx = cur->idx;
				ptr->w = cur->w;
				
				stack[stack_sz++] = ptr;
			}
			cur = cur->next;							
		}

	}
	free(stack);
	free(visited);
}


//input: a tree is defined by a set of (n-1) edges: {..., {i, j, w_ij}, ...}
//output: a tree in the form of linked nodes, and the root node is labeled.
int transform(struct Edge *tree, int *nodes, int n_nodes, struct Node **root, int *y, int total)
{
	int idx = -1;
	for(int i=0; i<n_nodes; i++)
	{
		if(y[nodes[i]]>=0){
			idx = nodes[i];
			break;
		} 
	}
	if( idx==-1 ) return 0;

	struct Edge_ *pool = (struct Edge_ *)malloc(2*(n_nodes-1)*sizeof(struct Edge_));
	struct Edge_ **g = (struct Edge_ **)malloc(total*sizeof(struct Edge_ *));
	build_adj_graph(tree, total, n_nodes-1, g, pool);

	build_tree(g, n_nodes, idx, root, total);

	free(pool);
	free(g);
	return -1;
}


void free_tree(struct Node *root)
{
	if( !root ) return;

	struct Node *cur = root->first_chi;
	struct Node *next;
	while(cur){
		next = cur->next_bro;
		free_tree(cur);
		cur = next;
	}
	free( root );
}

void label_pure_tree(struct Node *root, int c, int *y)
{
	if( !root ) return;
	
	y[root->idx] = c;
	struct Node *cur = root->first_chi;
	while(cur){
		label_pure_tree(cur, c, y);
		cur = cur->next_bro;
	}
}

//Note: r MUST be an internal node
/*
void compute_lbtree_cutsize(struct Node *r, int n_class, float **cutsize, int *y)
{
	int r_idx, k;

	r_idx = r->idx;
	if( !(r->first_chi) ){//leaf node which is labeled.
		for( k=0; k<n_class; k++ ) cutsize[k][r_idx] = INFI;
		cutsize[y[r_idx]][r_idx] = 0;
		return;
	}

	for( k=0; k<n_class; k++ ) cutsize[k][r_idx] = 0;

	struct Node *cur = r->first_chi;
	while(cur){
		compute_lbtree_cutsize(cur, n_class, cutsize, y);
		
		int idx = cur->idx;
		for( k=0; k<n_class; k++ ){

			float c = cutsize[k][idx];
			for( int i=0; i<n_class; i++ ){
				if( c > cutsize[i][idx] + cur->w ){
					c = cutsize[i][idx] + cur->w;
				}
			}
			cutsize[k][r_idx] += c;
		}

		cur = cur->next_bro;
	}
	return;
}
*/


//Note: r MUST be an internal node
void compute_lbtree_cutsize(struct Node *r, int n_class, float **cutsize, int *y, int total)
{
	bool *visited = (bool *)malloc(total*sizeof(bool));
	memset(visited, 0, total*sizeof(bool));

	struct Node **stack = (struct Node **)malloc(total*sizeof(struct Node *));
	int stack_sz = 0;
	stack[stack_sz++] = r;

	while( stack_sz ){
		struct Node *ptr = stack[stack_sz-1];
		if( !visited[ptr->idx] ){
			visited[ptr->idx] = true;
			ptr = ptr->first_chi;
			while(ptr){
				stack[stack_sz++] = ptr;
				ptr = ptr->next_bro;
			}						
		}else{
			int k;
			int idx = ptr->idx;
			if( !ptr->first_chi ){
				for( k=0; k<n_class; k++ ) cutsize[k][idx] = INFI;
				cutsize[y[idx]][idx] = 0;
			}else{
				struct Node *cur = ptr->first_chi;
				while(cur){
					int cidx = cur->idx;
					for( k=0; k<n_class; k++ ){
						float c = cutsize[k][cidx];
						for( int i=0; i<n_class; i++ ){
							if( c > cutsize[i][cidx] + cur->w ){
								c = cutsize[i][cidx] + cur->w;
							}
						}
						cutsize[k][idx] += c;
					}
					cur = cur->next_bro;
				}				
			}
			stack_sz--;
		}
	}

	free(stack);
	free(visited);
}

/*
void find_minicut_labeling(struct Node *r, int n_class, float **cutsize, int *y, int parent)
{
	if( !(r->first_chi)) return; //leaf

	int c, idx;
	float m;
	idx = r->idx;
	if(parent==-1){//root
		c = 0;
		m = cutsize[0][idx];
		for(int i=1; i<n_class; i++)
			if( cutsize[i][idx]<m ){
				c = i;
				m = cutsize[i][idx];
			}
	}else{
		c = parent;
		m = cutsize[parent][idx];
		for(int i=0; i<n_class; i++)
			if( cutsize[i][idx] + r->w < m ){
				c = i;
				m = cutsize[i][idx] + r->w;
			}		
	}
	
	y[idx] = c;

	struct Node *cur=r->first_chi;
	while(cur){
		find_minicut_labeling(cur, n_class, cutsize, y, c);
		cur = cur->next_bro;
	}
	return;
}
*/

void find_minicut_labeling(struct Node *r, int n_class, float **cutsize, int *y,  int total)
{
	int c, idx;
	float m;

	idx = r->idx;
	c = 0;
	m = cutsize[0][idx];
	for(int i=1; i<n_class; i++)
		if( cutsize[i][idx]<m ){
			c = i;
			m = cutsize[i][idx];
		}
	y[idx] = c;

	int stack_sz = 0;
	struct Node **stack = (struct Node **)malloc(total*sizeof(struct Node *));
	stack[stack_sz++] = r;

	while( stack_sz ){
		struct Node *ptr = stack[--stack_sz];
		struct Node *chi = ptr->first_chi;
		while( chi ){
			stack[stack_sz++] = chi;
			c = y[ptr->idx];
			idx = chi->idx;
			m = cutsize[c][idx];
			for(int i=0; i<n_class; i++)
				if( cutsize[i][idx] + chi->w < m ){
					c = i;
					m = cutsize[i][idx] + chi->w;
				}
			y[idx] = c;
			
			chi = chi->next_bro;
		}
	}

	free( stack );
}

void treecut(struct Node *tree, int total, int n_class, int *rlts, float **cutsize)
{		
	//b. decompose the MST into lb-trees and pure-trees
	tree_split(tree, rlts, total);

	//c. label the sub-trees
	int i;
	for( i=0; i<n_lb; i++ ){
		struct Node *swp = lb_trees[i];
		lb_trees[i] = lb_trees[i]->first_chi;
		swp->first_chi = NULL;
		swp->next_bro = lb_trees[i]->first_chi;
		swp->w = lb_trees[i]->w;
		lb_trees[i]->first_chi = swp;

		compute_lbtree_cutsize(lb_trees[i], n_class, cutsize, rlts, total);
		find_minicut_labeling(lb_trees[i], n_class, cutsize, rlts, total);
		free_tree(lb_trees[i]);

	}
	for( i=0; i<n_pure; i++){
		int c = rlts[pure_trees[i]->idx];
		label_pure_tree(pure_trees[i], c, rlts);
		free_tree(pure_trees[i]);
	}
}

void majority_vote(int n_node, int n_class, int n_tree, int **rlts, int *y)
{
	int i, j, cmax;

	bool *is_labeled = (bool *)malloc(n_node*sizeof(bool));
	for( i=0; i<n_node; i++ ) is_labeled[i] = (rlts[0][i]==-1) ? false:true;
	int *hist = (int *)malloc(n_class*sizeof(int));
	for( i=0; i<n_node; i++ ){
		if( !is_labeled[i] ){
			y[i] = -1;
			continue;
		}
		memset(hist, 0, n_class*sizeof(int));
		for( j=0; j<n_tree; j++ ){
			hist[rlts[j][i]]++;
		}
		cmax = 0;
		for( j=1; j<n_class; j++ ){
			if( hist[j]>hist[cmax] ) cmax = j;			
		}
		y[i] = cmax;
	}

	free(hist);
	free(is_labeled);
}

void mtc(struct Edge *graph, int n_nodes, int n_edges, int *y, int n_class, int tree_type, int n_trees)
{
    printf("in-------");
	int i, j, *buf, **rlts;
	rlts = (int **)malloc(n_trees*sizeof(int *));
	buf = (int *)malloc(n_trees*n_nodes*sizeof(int));
	if( !rlts || !buf ){	
		printf("error in allocating memory.\n");
		exit(1);
	}
	for( i=0; i<n_trees; i++ ){
		rlts[i] = buf + i*n_nodes;
		memcpy(rlts[i], y, n_nodes*sizeof(int));
	}

	struct Edge *e_com;
	int n_components, *e_com_sz, *n_com, *n_com_sz;	
	e_com = (struct Edge *)malloc(n_edges*sizeof(struct Edge));
	e_com_sz = (int *)malloc(3*n_nodes*sizeof(int));
	if( !e_com || !e_com_sz ){	
		printf("error in allocating memory.\n");
		exit(1);
	}
	n_com = e_com_sz + n_nodes;
	n_com_sz = n_com + n_nodes;

	n_components = detect_disconnected_components(graph, n_nodes, n_edges, e_com, e_com_sz, n_com, n_com_sz);
	if( tree_type==0 || tree_type==1 ) 
		for(i=0; i<n_edges; i++) e_com[i].w = -e_com[i].w;

	float **cutsize=(float **)malloc(n_class*sizeof(float *));
	float *p = (float *)malloc(n_class*n_nodes*sizeof(float));
	if( !cutsize || !p ){	
		printf("error in allocating memory.\n");
		exit(1);
	}
	for(i=0; i<n_class; i++)
		cutsize[i] = p + i*n_nodes;

	bool *is_labeled = (bool *)malloc(n_components*sizeof(bool));
	for(i=0; i<n_components; i++) is_labeled[i] = false;
	int s = 0;
	for(i=0; i<n_components; i++){
		for(j=0; j<n_com_sz[i]; j++){
			if( y[n_com[s+j]] != -1 ){
				is_labeled[i] = true;
				break;
			} 
		}
		s += n_com_sz[i];
	}
	
	int ii, jj, pos1, pos2, r;
	for( jj=0; jj<n_trees; jj++ ){
	
		memset(p, 0, n_class*n_nodes*sizeof(float));
		
		pos1 = 0;
		pos2 = 0;	

		for( ii=0; ii<n_components; ii++ ){
			if( !is_labeled[ii] ){
				pos1 += e_com_sz[ii];
				pos2 += n_com_sz[ii];
				continue;
			} 
			//a. generate a spanning tree for component ii
			struct Edge *tree_graph = (struct Edge *)malloc((n_nodes-1)*sizeof(struct Edge));
			switch(tree_type){
				case 0:
					mini_spanning_tree(e_com+pos1, n_com+pos2, n_com_sz[ii], e_com_sz[ii], tree_graph, n_nodes);
					for(i=0; i<n_com_sz[ii]-1; i++) tree_graph[i].w = -tree_graph[i].w;
					break;
				case 1:
					r = floor(((double)rand())/((double)RAND_MAX)*n_com_sz[ii]);
					shortest_path_tree(e_com+pos1, n_com+pos2, n_com_sz[ii], e_com_sz[ii], tree_graph, n_nodes, n_com[pos2+r]);
					for(i=0; i<n_com_sz[ii]-1; i++) tree_graph[i].w = -tree_graph[i].w;
					break;
				case 2:
					random_spanning_tree(e_com+pos1, n_com+pos2, n_com_sz[ii], e_com_sz[ii], tree_graph, n_nodes);
					break;
				default:
					printf("unknown tree type\n");
					exit(1);
					break;	
			}
			
			struct Node *tree;
			int z = transform(tree_graph, n_com+pos2, n_com_sz[ii], &tree, rlts[jj], n_nodes);
			pos1 += e_com_sz[ii];
			pos2 += n_com_sz[ii];
			free(tree_graph);
			if(z==0){//no labeled node in this component
				printf("error in is_labeled[] array.\n");
				continue;
			}	
			//label this component
			//Note: the second parameter is @n_nodes, not @n_com_sz[ii].
			treecut(tree, n_nodes, n_class, rlts[jj], cutsize);
		}	
			
	}

	majority_vote(n_nodes, n_class, n_trees, rlts, y);

	free(is_labeled);
	free(e_com);
	free(e_com_sz);
	free(cutsize);
	free(p);
	free(buf);
	free(rlts);
}
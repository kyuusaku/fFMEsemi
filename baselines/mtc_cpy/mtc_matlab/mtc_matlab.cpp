#include "mex.h"
#include "mtc.h"

/******************* split.cpp ************************/
int n_tasks;
struct Node *tasks[MAX_ARRAY_LEN];//stack

int n_lb, n_pure;
struct Node *lb_trees[MAX_ARRAY_LEN];
struct Node *pure_trees[MAX_ARRAY_LEN];
int *pos;

#define LABELED	0
#define INTERNAL 1
#define EXTERNAL 2

struct Node *copy_node(struct Node *p)
{
	struct Node *c = (struct Node *)malloc(sizeof(struct Node));
	c->idx = p->idx;
	c->w = p->w;
	c->first_chi = p->first_chi;
	c->next_bro = p->next_bro;
	
	return c;
}

//input r: a tree that root at r which is a labeled node or an internal node
//output: return a lb-tree
struct Node *split(struct Node *r){
	struct Node *t, *cur;

	if( !r ) printf("error.");
	if( pos[r->idx]==LABELED ){
		if( !r->first_chi ){
			free(r);
			return NULL;
		}
		if(r->first_chi->next_bro){
			t = copy_node(r);
			t->first_chi = r->first_chi->next_bro;
			tasks[n_tasks++] = t; //push
		}
		cur = r->first_chi;
		cur->next_bro = NULL;
		
		if( pos[cur->idx]==LABELED ){
			free( r );
			tasks[n_tasks++] = cur;//push
			return NULL;
		}
		if( pos[cur->idx]==INTERNAL ){
			split( cur );
			return r;
		}
		if( pos[cur->idx]==EXTERNAL ){
			pure_trees[n_pure++] = r;
			return NULL;
		}
	}

	if( pos[r->idx]==1 ){
		cur=r->first_chi;

		while(cur){
			if( pos[cur->idx]==LABELED ){
				t=copy_node(cur);
				t->next_bro = NULL;
				tasks[n_tasks++] = t;
				cur->first_chi = NULL;
				cur = cur->next_bro;

			}else if( pos[cur->idx]==INTERNAL ){
				split(cur);
				cur = cur->next_bro;

			}else if( pos[cur->idx]==EXTERNAL ){ //pure tree
				t=copy_node(r);
				t->first_chi = cur;
				pure_trees[n_pure++] = t;

				//delete it from lb-tree
				t = r->first_chi;
				if( cur==t ){
					r->first_chi = cur->next_bro;
				}else{
					while(t->next_bro!=cur) t=t->next_bro;
					t->next_bro = cur->next_bro;
				}
				t = cur;
				cur = cur->next_bro;
				t->next_bro = NULL;
			}
		}
		return r;
	}	
}

//Note: root must be labeled.
void determine_node_position(struct Node *root, int *y, int total)
{
	bool *visited=(bool *)malloc(total*sizeof(bool));
	memset(visited, 0, total*sizeof(bool));

	int stack_sz=0;
	struct Node **stack = (struct Node **)malloc(total*sizeof(struct Node *));
	stack[stack_sz++] = root;

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
			if(y[ptr->idx]>=0) 
				pos[ptr->idx] = LABELED;
			else{
				pos[ptr->idx] = EXTERNAL;
				struct Node *cur = ptr->first_chi;
				while( cur ){
					if( pos[cur->idx]!= EXTERNAL ){
						pos[ptr->idx] = INTERNAL;
						break;
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

void tree_split( struct Node *r, int *y, int total)
{
	pos = (int *)malloc(total*sizeof(int));
	//NOTE: node *r MUST be labeled
	determine_node_position(r, y, total);

	n_lb = 0;
	n_pure = 0;
	n_tasks = 0;
	tasks[n_tasks++] = r;//push
//int cnt=0;
	while(n_tasks){
		struct Node *root;
		root = split(tasks[--n_tasks]);//pop

		if( root ) lb_trees[n_lb++] = root;
//printf("%d: n_lb: %d, n_pure: %d, n_tasks: %d\n", ++cnt, n_lb, n_pure, n_tasks);
	}

	free(pos);

}


/******************* random_spanning_tree.cpp ************************/

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

/******************* mini_spanning_tree.cpp ************************/

int cmp(const void *i, const void *j)
{
	if(((struct Edge *)i)->w < ((struct Edge *)j)->w) return -1;
	if(((struct Edge *)i)->w == ((struct Edge *)j)->w) return 0;
	
	return 1;
}

int *s;//union-find set
int *rank;

int find(int x)
{
	while( x!=s[x] ) x = s[x];		
	return x;
}

void Union(int x, int y)
{
	x = find(x);
	y = find(y);
	if(x==y) return;

	if(rank[x]>rank[y])
		s[y] = x;
	else{
		s[x] = y;
		if(rank[x]==rank[y]) rank[y]++;
	}
}


//input: total is the max num of node idx.
void mini_spanning_tree( struct Edge *graph, int *nodes, int n_nodes, int n_edges, struct Edge *tree, int total )
{
	int i, cnt;	

	qsort(graph, n_edges, sizeof(struct Edge), cmp);
	s = (int *)malloc(2*total*sizeof(int));
	rank = s + total;
	for(i=0; i<total; i++){
		s[i] = i;
		rank[i] = 0;
	}
	i = 0;
	cnt = 0;
	while(cnt<n_nodes-1){
		int x = find(graph[i].x);
		int y = find(graph[i].y); 
		if( x != y ){
			tree[cnt++] = graph[i];	
			Union(x, y);
		}
		i++;
	}

	free(s);
}

/******************* shortest_path_tree.cpp ************************/
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

/******************* detect_disconnected_components.cpp ************************/

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

/******************* mtc.cpp ************************/

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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    int n_nodes, n_edges, n_classes, n_trees, tree_type;
    
    double *ptr;
    
    ptr = mxGetPr(prhs[1]);
    n_nodes = (int)(ptr[0]);
    
    ptr = mxGetPr(prhs[3]);
    n_classes = (int)(ptr[0]);
    
    ptr = mxGetPr(prhs[4]);
    tree_type = (int)(ptr[0]);    
    
    ptr = mxGetPr(prhs[5]);
    n_trees = (int)(ptr[0]);    
        
    ptr = mxGetPr(prhs[0]);
    n_edges = mxGetM(prhs[0]);
    struct Edge *graph = (struct Edge *)malloc(n_edges*sizeof(struct Edge));
    for(int i=0; i<n_edges; i++ ){
        graph[i].x = (int)(ptr[i])-1;
        graph[i].y = (int)(ptr[n_edges+i])-1;
        graph[i].w = ptr[2*n_edges+i];
    }
    
    printf("# of nodes: %d, # of edges: %d\n", n_nodes, n_edges);
    printf("# of classses: %d\n", n_classes);
    if( tree_type==0 )
        printf("tree type: minimum spannining tree\n");
    else if( tree_type==1 )
        printf("tree type: shortest path tree\n");
    else if( tree_type==2 )
        printf("tree type: random spanning tree\n");                
    printf("# of tree: %d\n", n_trees);    
    
    ptr = mxGetPr(prhs[2]);
    int *y = (int *)malloc(n_nodes*sizeof(int));
    for(int i=0; i<n_nodes; i++)
        y[i] = (int)(ptr[i]);
 /**/   
    
    mtc(graph, n_nodes, n_edges, y, n_classes, tree_type, n_trees);
    
    plhs[0] = mxCreateDoubleMatrix(n_nodes, 1, mxREAL);
    ptr = mxGetPr(plhs[0]);
    for(int i=0; i<n_nodes; i++)
        ptr[i] = y[i];
    
    free(graph);
    free(y);        
    return;
}

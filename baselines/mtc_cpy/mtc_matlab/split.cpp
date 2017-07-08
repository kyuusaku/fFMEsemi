#include "mtc.h"

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
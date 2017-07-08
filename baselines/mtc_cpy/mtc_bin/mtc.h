#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"

#define INFI 100000000
#define MAX_ARRAY_LEN 1000000

#define LINE_LEN 1024

struct Node{
	int idx;
	float w;//weight between this node and its father
	struct Node *first_chi;
	struct Node *next_bro;
};

struct Edge{
	int x,y;
	float w;
};

struct Edge_{
	int idx;
	float w;
	struct Edge_ *next;
};

void load_graph(char *graph_file, int *n_node_p, int *n_edge_p, struct Edge **edges);
void load_labels(char *train_file, int n, int *n_classes, int **y, int **label_idx);
void output_rlts(char *rlts_file, int n, int *y, int *label_idx);


void build_adj_graph(struct Edge *graph, int n_nodes, int n_edges, struct Edge_ **g, struct Edge_ *pool);
void mini_spanning_tree( struct Edge *graph, int *nodes, int n_nodes, int n_edges, struct Edge *tree, int total );
void shortest_path_tree( struct Edge *graph, int *nodes, int n_nodes, int n_edges, struct Edge *tree, int total, int root );
void random_spanning_tree(struct Edge *graph, int *nodes, int n_nodes, int n_edges, struct Edge *tree, int total);

int detect_disconnected_components(struct Edge *graph, int n_nodes, int n_edges, struct Edge *e_com, int *e_com_sz, int *n_com, int *n_com_sz);
void tree_split( struct Node *r, int *y, int n_nodes);

void mtc(struct Edge *graph, int n_node, int n_edge, int *rlts, int n_class, int tree_type, int n_tree);
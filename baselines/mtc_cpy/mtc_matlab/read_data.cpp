
#include "mtc.h"

//假设node idx是从0开始的连续整数
void load_graph(char *graph_file, int *n_node_p, int *n_edge_p, struct Edge **edges)
{
	FILE *fd = fopen(graph_file, "r");
	if(!fd){
		printf("error in reading file: %s\n", graph_file);
		exit(1);
	}

	int n_node = -1;
	int n_edge = 0;
	int buf_sz = 0;
	int block_sz = 1000000;
	struct Edge *buf = NULL;
	char line[LINE_LEN];
	while( fgets(line, LINE_LEN, fd) )
	{
		if( n_edge == buf_sz ){
			buf_sz += block_sz;
			buf = (struct Edge *)realloc( buf, buf_sz*sizeof(struct Edge) );
			if( !buf ){
				printf("error in (re)allocating memory when load graph\n");
				exit(1);
			}
		}
		int x, y;
		float w;
		sscanf(line, "%d,%d,%f", &x, &y, &w);
		buf[n_edge].x = x;
		buf[n_edge].y = y;
		buf[n_edge++].w = w;

		if( n_node<x ) n_node = x;
		if( n_node<y ) n_node = y;
	}

	fclose(fd);
	*n_node_p = n_node+1;
	*n_edge_p = n_edge;
	*edges = buf;

	return;
}

//假设label是从0开始的连续整数
void load_labels(char *train_file, int n, int *n_classes, int **y, int **label_idx)
{
	FILE *fd = fopen(train_file, "r");
	if(!fd){
		printf("error in reading label file.\n");
		exit(1);
	}

	int *labels = (int *)malloc(2*n*sizeof(int));	
	if( !labels ){
		printf("error in allocating memory when load training file\n");
		exit(1);
	}
	for( int i=0; i<2*n; i++ ) labels[i] = -1;

	int K = -1;
	int idx, c;
	while( fscanf(fd, "%d:%d\n", &idx, &c)==2 ){
		labels[idx] = c;
		labels[n+idx] = 1;
		if( c>K ) K = c;
	}

	*y = labels;
	*label_idx = labels+n;
	*n_classes = K+1;
	fclose(fd);
	return;
}

void output_rlts(char *rlts_file, int n, int *y, int *label_idx)
{
	FILE *fd = fopen( rlts_file, "w" );
	for( int i=0; i<n; i++ )
		if(label_idx[i]==-1) fprintf(fd, "%d:%d\n", i, y[i]);

	fclose(fd);
}
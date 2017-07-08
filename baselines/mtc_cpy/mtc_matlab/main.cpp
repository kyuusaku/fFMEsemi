#include "mtc.h"


void exit_with_help()
{
	printf(
		"Useage: mtc [options] graph_file train_file output_file\n"
		"options:\n"
		"-t: select the type of spanning tree (default: 0)\n"
		"	0 -- minimum spanning tree\n"
		"	1 -- shortest path tree\n"
		"	2 -- random spanning tree\n"
		"-n: set the number of spanning trees (default: 1)\n"
		);
	exit(1);
}

void parse_command_line(int argc, char **argv, int *tree_type, int *n_tree, char *graph_file, char *train_file, char *output_file)
{
	int i;

	*tree_type = 0;
	*n_tree = 1;

	for( i=1; i<argc; i++ ){
	
		if(argv[i][0] != '-') break;
		if(++i >= argc) exit_with_help();

		switch(argv[i-1][1]){
		
			case 't':
				*tree_type = atoi(argv[i]);
				break;
			case 'n':
				*n_tree = atoi(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	
	if( i != argc-3 )
		exit_with_help();
	else{	
		strcpy(graph_file, argv[i]);
		strcpy(train_file, argv[i+1]);
		strcpy(output_file, argv[i+2]);
	}
}

int main(int argc, char **argv)
{
	int tree_type, n_tree, n_node, n_edge, n_class;
	struct Edge *graph;
	int *y, *label_idx;
	char graph_file[LINE_LEN], train_file[LINE_LEN], output_file[LINE_LEN];

	parse_command_line(argc, argv, &tree_type, &n_tree, graph_file, train_file, output_file);

	load_graph(graph_file, &n_node, &n_edge, &graph);
	load_labels(train_file, n_node, &n_class, &y, &label_idx);

	clock_t tstart, tend;
	tstart = clock();	
	mtc(graph, n_node, n_edge, y, n_class, tree_type, n_tree);
	tend = clock();
	printf("The running time: %.3f (seconds)\n", ((double)(tend-tstart))/CLOCKS_PER_SEC);

	output_rlts(output_file, n_node, y, label_idx);
	
	free(y);
	free(graph);

	return 0;
}
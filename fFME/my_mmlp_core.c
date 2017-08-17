/*
 *  my_mmlp_core.c
 *
 *
 *  compile
 *  -----------------------------------------------------------------------
 *  mex -largeArrayDims my_mmlp_core.c
 * 
 *  usage
 *  -----------------------------------------------------------------------
 *  my_mmlp_core(labeledset, G, f, v, thres)
 *
 *  labeledset:	N_L-by-1 vector
 *  labeledset(i):	1,...,N  the actual node index of the i-th labeled node
 *
 *  G:	N-by-N sparse matrix
 *  G(i,j):	> 0  the distance between x_i and x_j, if the edge (i,j) exists
 *			= 0  there is no edge between x_i and x_j
 *
 *  f:	N-by-1 vector
 *  f(i):	= 1,2,...,C  the true class label for x_i (labeled node)
 *			= 0		  x_i is unlabeled
 *  f(labeledset) should have their true class labels
 *
 *  v:	N-by-1 vector
 *  v(i):	the minimax distance from x_i to its nearest labeled node
 *  v(labeledset) should be 0
 *
 *  thres:	a value between 0 and 1
 *			it controls how early MMLP should terminate
*/

#include "mex.h"

#define EARLY_STOPPING
#define MEASURE_PERFORMANCE

#define NUM_ARG_IN		4
#define ARY_SEED		prhs[0]
#define SPMAT_GRAPH		prhs[1]
#define ARY_F			prhs[2]
#define ARY_V			prhs[3]

#ifdef MEASURE_PERFORMANCE
	#define NUM_ARG_OUT		3
	#define VAL_ITER_STOP	plhs[0]
	#define ARY_SIZE_Q		plhs[1]
	#define ARY_HOP			plhs[2]
	#define MAX_ITER		10000
#endif

#ifdef EARLY_STOPPING
	#define VAL_STOP_THRESHOLD		prhs[4]
#endif

void mexFunction( int nlhs, mxArray *plhs[],
				  int nrhs, const mxArray *prhs[] )
{
	size_t N;  /* the number of nodes */
	size_t N_L;  /* the number of labeled nodes */

	size_t *Q;  /* a circular queue of nodes waiting for propagation */
	size_t Q_max_length, Q_first, Q_last;  /* max, first, last indices in Q*/
	bool Q_not_empty;
	bool *is_in_Q;  /* to indicate whether each node is in Q */

	double *labeled;  /* vector of indices of labeled nodes */
	double *f;  /* vectors of predicted labels and */
	double *v;  /* the corresponding minimax distances */
	size_t i, j;

	double *G;  /* the graph of N nodes, given by an N-by-N sparse matrix */
	size_t *ir, *jc;  /* and some variables to access the entries in G */
	size_t t, tt;

	double val;  /* temporary variable */

#ifdef EARLY_STOPPING
	size_t stop_threshold;
	bool *has_visited;
	size_t num_has_visited;
#endif

#ifdef MEASURE_PERFORMANCE
	size_t num_iter;  /* iteration count */
	size_t Q_next_first;  /* to check whether the current iteration has done */
	double *Q_size;  /* vector of the size of Q at each iteration */
	double *hop;  /* vector of the number of hops of each node */
#endif

	/* check command error */
	if (nrhs < NUM_ARG_IN)
		mexErrMsgTxt("Incorrect number of input arguments.");
	if (nlhs > NUM_ARG_OUT)
		mexErrMsgTxt("Too many output arguments");

	/* get the number of nodes */
	N = mxGetN(SPMAT_GRAPH);
	N_L = mxGetNumberOfElements(ARY_SEED);

	/* allocation and initializion */
	f = mxGetPr(ARY_F);
	v = mxGetPr(ARY_V);
	Q = mxMalloc(N*(sizeof *Q));
	is_in_Q = mxCalloc(N, sizeof *is_in_Q);

#ifdef EARLY_STOPPING
	has_visited = mxCalloc(N, sizeof *has_visited);
	num_has_visited = N_L;
	if (nrhs == NUM_ARG_IN+1)
		stop_threshold = (size_t) (N*mxGetScalar(VAL_STOP_THRESHOLD));
	else
		stop_threshold = (size_t) N;
#endif

	/* push all N_L labeled nodes into Q
	 N_L labeled nodes are initially inserted into Q as seeds for propagation */
	labeled = mxGetPr(ARY_SEED);
	for (t = 0; t < N_L; ++t) {
		i = (size_t)labeled[t]-1;
		Q[t] = i;
		is_in_Q[i] = true;

#ifdef EARLY_STOPPING
		has_visited[i] = true;
#endif
	}
	Q_max_length = N-1;
	Q_first = 0;  /* labeled nodes are now in Q, from Q[0] to Q[N_L-1] */
	Q_last = N_L-1;
	Q_not_empty = true;  /* thus Q is non-empty */

#ifdef MEASURE_PERFORMANCE
	VAL_ITER_STOP = mxCreateDoubleMatrix(1,1, mxREAL);
	num_iter = 0;

	ARY_SIZE_Q = mxCreateDoubleMatrix(MAX_ITER,1, mxREAL);
	Q_size = mxGetPr(ARY_SIZE_Q);
	Q_size[0] = (double) N_L;
	Q_next_first = N_L; /* iteration 1 will be done after visiting L labeled nodes */

	ARY_HOP = mxCreateDoubleMatrix(N,1, mxREAL);
	hop = mxGetPr(ARY_HOP);
#endif

	/* perform minimax label propagation */
	G = mxGetPr(SPMAT_GRAPH);
	ir = mxGetIr(SPMAT_GRAPH);
	jc = mxGetJc(SPMAT_GRAPH);
	while (Q_not_empty)
	{
		/* pop a node (say node i) from Q to propagate the label of the node */
		if (Q_first == Q_last) {
			Q_not_empty = false;
		}
		if (Q_first < Q_max_length) {
			i = Q[Q_first++];
		}
		else {
			Q_first = 0;
			i = Q[Q_max_length];
		}

		/* visit each adjacent node of node i (say node j)
		 to propagate the label assigned to node i into node j */
		tt = jc[i+1];
		for (t = jc[i]; t < tt; ++t) {
			j = ir[t];
            
            mexPrintf("i: %d j: %d", i, j);

			/* check whether node j's label and minimax distance
			 should be updated or not */
			/*val = max(v[i], G[t]); */
            if (v[i] >= G[t])
            {
                val = v[i];
            }
            else
            {
                val = G[t];
            }
            
			if (val >= v[j])
				continue;

			/* propagate node i's label into node j
			 and update the corresponding minimax distance */
			v[j] = val;
			f[j] = f[i];

#ifdef MEASURE_PERFORMANCE
			/* count the number of hops from the source */
			hop[j] = (double) num_iter;
#endif

			/* push this updated node j into Q for further propagation
			 unless j is already in Q */
			if (!is_in_Q[j]) {
				if (Q_last < Q_max_length) {
					Q[++Q_last] = j;
				}
				else {
					Q_last = 0;
					Q[0] = j;
				}
				is_in_Q[j] = true;

#ifdef EARLY_STOPPING
				if (!has_visited[j]) {
					has_visited[j] = true;
					++num_has_visited;
				}
#endif
			}
		}

		/* now node i is not in Q */
		is_in_Q[i] = false;

#ifdef MEASURE_PERFORMANCE
		/* check whether the current iteration has finished */
		if (Q_first == Q_next_first) {
			++num_iter;

			/* reset Q_next_first to Q_last+1 */
			if (Q_last < Q_max_length)
				Q_next_first = Q_last+1;
			else
				Q_next_first = 0;

			/* count the size of Q for the next iteration */
			if (Q_next_first >= Q_first)
				Q_size[num_iter] = (double) (Q_next_first - Q_first);
			else
				Q_size[num_iter] = (double) ((Q_max_length+1 - Q_first) + Q_next_first);
		}
#endif

#ifdef EARLY_STOPPING
		/* when early-stopping is used, terminate the algorithm earlier if
		 most nodes (e.g. 90%) have visited once or more */
		if (num_has_visited >= stop_threshold) {
#ifndef MEASURE_PERFORMANCE
			break;
#else
			/* store the iteration count at that time */
			*mxGetPr(VAL_ITER_STOP) = num_iter;

			/* will never visit here again */
			stop_threshold = (size_t) (N+1);
#endif
		}
#endif
	}

#ifdef MEASURE_PERFORMANCE
	/* when the algorithm converge */
	if (!Q_not_empty) {
		/* store the iteration count at convergence */
		*mxGetPr(VAL_ITER_STOP) = num_iter;
	}
#endif

	mxFree(Q);
	mxFree(is_in_Q);

	return;
}
Code for minimum tree cut for fast semi-supervised classification
Copyright 2014 Yan-Ming Zhang (ymzhang@nlpr.ia.ac.cn)


******************************************************************
This software is currently released under the GNU General Public License 
http://www.gnu.org/copyleft/gpl.html

If you use this code in research for publication, please cite our TNNLS 
paper "MTC: A Fast and Robust Graph-Based Transductive Learning Method". 

Please provide feedback if you have any questions or suggestions via 
ymzhang@nlpr.ia.ac.cn.
*******************************************************************

 
0. Installation
==================

mex mtc_matlab.cpp ..\mtc_bin\mtc.cpp ..\mtc_bin\detect_disconnected_components.cpp 
..\mtc_bin\mini_spanning_tree.cpp ..\mtc_bin\shortest_path_tree.cpp 
..\mtc_bin\random_spanning_tree.cpp ..\mtc_bin\split.cpp


1. Usage
==================


f = mtc_matlab(edges, n, y, k, tree_type, n_tree);


f : a n-by-1 vector that contains the predicted labels for all nodes. 
edges : a m-by-3 matrix in which each row [idx1,idx2,weight] is one edge in the graph. The node index must be continuous integers and start from 1.
n : the number of nodes in the graph.
y : a n-by-1 vector in which  y_i is the label of node i for labeled nodes, otherwise y_i=-1.
k : the number of classes.
tree_type : the type of spanning tree (0 -- minimum spanning tree, 1 -- shortest path tree, 2 -- random spanning tree).
n_tree : number of spanning trees.



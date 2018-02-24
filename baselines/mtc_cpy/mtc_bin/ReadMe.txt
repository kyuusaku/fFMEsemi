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
===============


1. Usage
========
	Useage: mtc [options] graph_file train_file output_file
	options:
	-t: select the type of spanning tree (default: 0)
		0 -- minimum spanning tree
		1 -- shortest path tree
		2 -- random spanning tree
	-n: set the number of spanning trees (default: 1)


2. Data format
==============

graph_file:
idx1,idx2,weight
.
.
.
Each line contains an edge and is ended by a '\n' character. The node index must be continuous integers and start from 0.

label_file:
idx:label
.
.
.
Each line contains the index of an labeled node and its label. The labels must be continuous integers and start from 0.

output_file:
idx:label
.
.
.
Each line contains the index of an unlabeled node and its predicted label. The labels must be continuous integers and start from 0.


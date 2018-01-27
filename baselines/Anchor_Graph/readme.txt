
Please first see usps_demo.m to find how my codes work.

To run AnchorGraph.m,  one needs to input anchors. In my ICML'10 paper, I used K-means clustering centers 
as anchors. If one had any sophisticated or task specific clustering algorithms, it could be better to feed 
the resulting clustering centers to anchors. Nevertheless, I found K-means anchors are sufficiently good.

Another possible issue is dimensionality. I strongly suggest users to conduct dimension reduction such
as PCA or LSA before running AnchorGraph.m. The proper dimension for data on which AnchorGraph is to be
constructed is 100-1000.

For any problem with my codes, feel free to drop me a message via wliu@ee.columbia.edu. Also, I hope you
to cite my ICML'10 paper in your publications.

Wei Liu
April 18, 2011

  






 
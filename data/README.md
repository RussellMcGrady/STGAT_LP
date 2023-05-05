# How to code Graph attention networks for link prediction using time series data in PyTorch
# do not revise the name of the headline in each csv.

Data files are three-fold FYI:

1. edge.csv should comprise the edge list in the graph, where the "startIdx" and "endIdx" are required.

2. forecast_data.csv should comprise all the attributes of nodes.
   Note that the "Node Index", "Timestamp", "Node Label", "Node" are required.
   All nodes should have the same number of periods, and in an order like the data format given in thes example.

3. projection_map.csv specify the number of features for nodes in each class/label.
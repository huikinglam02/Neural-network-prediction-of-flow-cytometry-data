# Neural-network-prediction-of-flow-cytometry-data

This is a simple application of neural network classification to flow cytometry data. The data was from this article:
Sachs, Karen, et al. "Causal protein-signaling networks derived from multiparameter single-cell data." Science 308.5721 (2005): 523-529.

In the data, fluorescence of 11 colors were recorded for T cells activated in different manners, for example with certain molecular processes inhibited. The fluorescence data were of various molecular markers of activation, detailed in the article. This comes to the following question: given the known perturbation condition, could one predict from the 11 color single cell data what condition was the cell treated?

I used neural network classification code from Andrew Ng's coursera class to conduct machine learning. The network used was of one hidde layer with 11 hidden units (the same number as the input) and the output was of 13 units, namely the number of conditions. After training and cross-validation, the overall classification accuracy was about 65 %, although when looking into accuracy of individual conditions, some conditions showed remarkably high accuracy around 80 %-90% (e.g. ones with PKC inhibitor G0076), whereas ones involving the PI3K inhibitior LY-294002 was particularly bad. 

Interestingly the conditions showing good prediction accuracy are the ones in which the cells are treated with specific inhibitors against the measured molecules,  hinting that those cells might show a much more distinct molecular activation profile that the neural network can easily pick up.

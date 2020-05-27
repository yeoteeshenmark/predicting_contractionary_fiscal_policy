This code uses Gaussian RBF Kernel Support Vector Machine (SVM) to predict if the US government is going to reduce spending over the next quarter with latest economic data.

As of the date of uploading the code, the model is able to predict with 69.2% accuracy whether the US government will reduce its spending. With out-of-sample testing, the model could correctly guess 9 of the 13 quarters the action taken by the US government in regards to government spending.

The explanatory variables in the model are foreign-debt-to-GDP, 10-yr government bond yield, trade weighted U.S. Dollar Index (representing the strength of US currency relative to world curriencies), and real GDP growth rate. All of the variables are in % change quarter-to-quarter. Thus the data-set for the kernel SVM model is 4-dimensional. I have lagged the variables in an attempt to be ahead of markets.

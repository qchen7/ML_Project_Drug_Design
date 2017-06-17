# ML_Project_Drug_Design
Machine learning project for drug design

I. MODEL TESTING
TO test the models for classification and regression, simply run
classification.py and regression.py.
There will be a simple user interface in the console shown similar to
the following part:
=========================================================
choose the model for testing:
1. softmax
2. ann
3. cnn
==========================================================
Simply input a number (1-3) to test the corresponding model.
II. GLOBAL PARAMETERS SETTINGS
TO change the gloabl parameters such as number of training epoches, go
to the following part of the code:
==========================================================
# gloab parameters settings, change training epochs here
total_len = x_train.shape[0]
n_input = x_train.shape[1]
disp_step = 10
training_epochs = 10000
learning_rate = 0.001
batch_size = total_len
==========================================================
To perform L2 regularization, find the content between following
comments:
    ##=====Uncomment below to perform L2 Regularization=====##
    ##...
    ##...
    ##=====Uncomment above to perform L2 Regularization=====##
Simply uncomment between to run
For example, ann is typically much slower than the other two models,
one may want to set "training_epochs = 500"
to see the plot of training cost vs validation cost or plot regression
results more quickly.
*NOTE:
For regression test, I use the myFP_CHEMBL217_trans.csv (which is a
transformation of myFP_CHEMBL217 to
stucture dataset using excel and R, with ~5000 records), since the
"target" for all records are the same,
which potentially lower the difficulty of the problem.
For classification test, I use myFP_217_D2.csv (with ~8000 records),
which contains 7 different "targets"
for classification. The preprocessing part (transform original data to
stucture format for python to use) is
credited to Xueli (Jason) Zhou, which can be used if new data has the
same format as the original data. My excel
and R combined transformation is not convenient to be implemented.

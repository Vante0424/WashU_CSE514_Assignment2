# WashU_CSE514_Assignment2
CSE514 – Spring 2023 Programming Assignment 2

This assignment is to give you hands-on experience with dimension reduction and the comparison of different classification models. It consists of a programming assignment (with optional extensions for bonus points) and a report. This project can be done in groups up to three, or as individuals. Feel free to use the Piazza team-search function to help find group-members

Topic

Compare, analyze, and select a classification model for identifying letters in various fonts.

Programming work

A) Data preprocessing This dataset contains 26 classes to separate, but for this assignment, we’ll simplify to three binary classification problems.

Pair 1: H and K Pair 2: M and Y Pair 3: Your choice For each pair, set aside 10% of the relevant samples to use as a final validation set.

Optional extension 1 – Multi-class classification For 10 bonus points, implement multi-class classification using all 26 classes.

B) Model fitting For this project, you must pick 2*(size of group) from the following classification models:

1. k-nearest neighbors 4. SVM

2. Decision tree 5. Artificial Neural Network

3. Random Forest 6. Naïve Bayes Classifier

For each model, choose a hyperparameter to tune using 5-fold cross-validation. You must test at least 3 values for a categorical hyperparameter, or at least 5 values for a numerical one. Hyperparameter tuning should be done separately for each classification problem; you might end up with different values for classifying H from K than for classifying M from Y.

Optional extension 2 – Tune more hyperparameters For bonus points, tune more than just one hyperparameter per model.

1 bonus point for each additional hyperparameter, up to 5 bonus points total. Optional extension 3 – Consider more classification models For bonus points, use/suggest additional classification models to me.

If I give the go-ahead, include one for 5 bonus points or two for 10 bonus points.

C) Dimension reduction For each of the models, implement a method of dimension reduction from the following:

1. Simple Quality Filtering 4. Embedded Methods

2. Filter Methods 5. Feature Extraction

3. Wrapper Feature Selection

Use your chosen method(s) to reduce the number of features from 16 to 4. Retrain your models using reduced datasets, including hyperparameter tuning. IMPORTANT: You may use any packages/libraries/code-bases as you like for the project, however, you will need to have control over certain aspects of the model that may be black-boxed by default. For example, a package that trains a kNN classifier and internally optimizes the k value is not ideal if you need the cross-validation results of testing different k values.

Data to be used

We will use the Letter Recognition dataset in the UCI repository at

UCI Machine Learning Repository: Letter Recognition Data Set (https:// archive.ics.uci.edu/ml/datasets/letter+recognition) Note that the first column of the dataset is the response variable (i.e., y).

There are 20,000 instances in this dataset.

For each binary classification problem, first find all the relevant samples (ex. all the H and K samples for the first problem). Then set aside 10% of those samples for final validation of the models. This means that you cannot use these samples to train your model parameters, your model hyperparameters, or your feature selection methods.

What to submit – follow the instructions here to earn full points

• (80 pts total + 21 bonus points) The report o Introduction (20 pts + 2 bonus points) § (4 pts) Your description of the problem and the practical impacts of solving it.

§ (4 pts) Motivation for multiple classifiers. What factors should be considered in determining a classifier as the “best,” e.g. computational complexity, validation accuracy, model interpretability, etc.

§ (4 pts) Motivation for dimension reduction. Which methods are “better,” and what factors should be considered in determining a dimension reduction method as “good” or “bad.”

§ (4 pts) Brief description of the dimension reduction method(s) you chose.

§ (4 pts) Speculate on the binary classification problems. Which pair of letters did you choose for the third problem? Which pair do you predict will be the easiest or hardest to classify?

§ (+2 bonus points) Discuss the advantages/disadvantages of using a multi-class classifier instead of a set of binary classifiers.

o

Results (36 pts + 13 bonus points) § For each classifier:

• (6/3/2 pts) Brief description of the classifier and its general advantages and disadvantages.

• (6/3/2 pts) Figure: Graph the cross validation results (from fitting the classification model without dimension reduction) over the range of hyperparameter values you tested. There should be three sets of values, one for each binary classification problem. • (6/3/2 pts) Figure: Graph the cross validation results (from fitting the classification model with dimension reduction) over the range of hyperparameter values you tested. There should be three sets of values, one for each binary classification problem.

o

§ (+5 bonus points) for additional hyperparameters tuned § (+4 bonus points) for additional classifiers § (+4 bonus points) for multi-class classification Discussion (19 pts + 6 bonus points) § (5 pts) Compare the performance and run time of the different classifiers on the final validation sets with either a table or a figure.

§ (5 pts) Compare the performance and run time of the different classifiers after dimension reduction on the final validation sets with either a table or a figure.

§ (9 pts) Lessons learned: What model would you choose for this problem and why? How did dimension reduction effect the accuracy and/or run times of the different classifiers? What would you do differently if you were given this same task for a new dataset? Include at least one additional topic of discussion.

§ (+4 bonus points) for including the additional classifiers in your discussion § (+2 bonus points) for including the multi-class classifier in your discussion

• (25 pts total + 4 bonus points) Your code (in a language you choose) including: o (5 pts) Instructions for running your code o (5 pts) The code for processing the data into training/testing sets o (10 pts) The code for your classifiers o (5 pts) The code for your dimension reduction method o (+2 bonus points) for additional classifiers o (+2 bonus points) for multi-class classifier

Due date

Monday, April 10 (midnight, STL time). Submission to Gradescope via course Canvas.

A one week late extension is available in exchange for a 20% penalty on your final score.

About the extra credit:

The bonus point values listed are the upper limit of extra credit you can earn for each extension. How many points you get will depend on how well you completed each task. Feel free to include partially completed extensions for partial extra credit!

In total, you can earn up to 25 bonus points on this assignment, which means you can actually get a 125% as your score if you submit it on time, or you can submit this assignment late with the 20% penalty and still get a 100% as your score. It’s up to you how you would prefer to manage your time and effort.

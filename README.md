# Data-Mining

Abstract

The marketing environment is very competitive.
The growing amount of data, customer changing
behavior, and technological advancement have
forced the business industries, including the banking
sector, to change their strategy to make it more
efficient and stay competitive. Data mining could
help the bank to look for a hidden pattern and predict
the marketing outcome. This report aims to
apply data mining classification techniques Logistic
Regression, Naive Bayes, Decision Tree C5.0,
and Random Forest to the real-world direct marketing
campaign dataset to make the campaign
more efficient by predicting the customer’s subscription
to the term deposit.
The Experiments conducted show the success of
the data mining application. The model’s performance
is further evaluated using statistical measures
such as accuracy, Precision, Recall, etc., and
the K fold cross-validation technique, which further
identifies the model that predicts the outcome
with high accuracy.

Introduction:

Since the massive transformation from manual to digital
in the banking sector, a significant amount of data is produced
daily via various transactions such as ATMs and
online credit cards. The information collected includes customer
details, transactions, credit cards, deposits, loans, and
the details of other services the bank offers. Data mining
helps to identify valid novel hidden patterns and valuable
insights in the large amount of data that traditional mathematical
and statistical techniques could not detect (Chitra
& Subashini, 2013). The bank operates with the investment
done by the customers. In return, the banks provide ser-
vices to customers to increase their profit and revenue, and
the money invested by the customers is returned with the
amount gained through the interest rate. Customer relationship
management is a crucial factor for the banking sector.
Many banks’ challenges are retaining the most profitable
customers and preventing them from moving to the competitors.
Customer relation management focuses on customer
retention, and acquisition (Moin & Ahmed, 2012). The data
mining gives the credit score for each customer based on
different factors credit history, past account due dates and
payment date, salary, age, etc(Sadatrasoul et al., 2013). It
predicts the customers who might default the payment or
will not repay the loan.
Fraud prevention and detection are other significant issues
banks face. Data mining applications for fraud prevention
in the bank follow the third-party data warehouse and cross
reference with its database for signs of the internal problem
(Farooqi et al., 2017). The data mining application on the
Customer segmentation classifies the customers based on
the various attributes having similar characteristics or buying
preferences to create a custom-made loyalty campaign
just for that customer segment. The profitability and productivity
increased by identifying the customers responding
to the services(Sing’oei & Wang, 2013). In this report, the
data mining classification techniques such as Logistic Regression(
LR), Decision Tree(DT), Naive Bayes(NB), and
Random Forest(RF) are applied to the bank’s direct marketing
data set to predict the customer subscribe to the term
deposit.

Data Mining Techniques.

The Logistic Regression, C5.0 Decision Tree, Naive Bayes
and random Forest are the data mining classification techniques
used in this report.The description of each of the
techniques are as below.

5.1. Logistic Regression:

The logistic regression identifies the association between
the dependent and a set of independent fields. Logistic
regression is used when the dependent variable has only
two values. In the direct marketing data used in this report,
the dependent variable y has only two values,Yes or No
Hence this is considered one of the best techniques to predict
the outcome of the direct marketing campaign(Elsalamony,
2014).

5.2. C5.0 Decision Tree:

A decision tree is the supervised machine learning algorithm
for classification and regression problems. The decision tree
functions on both categorical and continuous input and output
variables. The decision trees build classification in the
form of the tree structure and spit the data into subsets. The
association subtree developed incrementally, the node denotes
the test, and the branch is the outcome or terminal
node. Moreover, the leaf node is where the process is recursive
till all the nodes contain the same class(Elsalamony,
2014). Among the other Decision tree algorithms, C5.0 used
in this report is proven to give better accuracy than other algorithms.
C5.0 gives a binary decision tree or multi branches
tree. In C5.0, the attribute selection criteria are based on the
Information gain or Entropy to build a decision tree. The
C5.0 adopts the binomial confidence limit method as a pruning
technique to reduce the tree size without compromising
on its accuracy. While handling the missing values, the C5.0
distribute value probability among the outcomes(Sing’oei &
Wang, 2013).
The Decision tree model was initially applied to the train
data set. The model gets trained using the dataset containing
the outcome’s history. Afterward, the model is applied to
the new test dataset to predict the outcome. The figure 1
shows the process.
Figure 1. Decision Tree Model

5.3. Naive Bayes:

Naive Bayes is the probabilistic classifiers algorithm. Bayes
theorems focus on different conditional probabilities(Chitra
& Subashini, 2013). Na¨ıve Bayes is based on the collection
of the Bayes Theorem. It is also called as the independent
conditional probability; The conditional probability is the
outcome of the likelihood of the sample belonging to the
class. If the probability is 0, then there is no chance for the
event to occur. While the probability is one, there is a 100
percent chance for the event to occur. Since it simplifies
the computational effort by simply classifying each pair
of features independent of others, it is called Na¨ıve. The
Na¨ıve Bayes algorithm first converts the dataset into the
frequency table. Then, the likelihood table is created based
on the probabilities identified from the frequency table. The
posterior probability for each class is calculated based on
the Na¨ıve Bayesian equation. The class with the highest
posterior probability is the outcome of the prediction. The
posterior probability is calculated as shown in the below
figure 2.

5.4. Random Forest:

The Random Forest is the most flexible machine learning
method capable of performing regression and classification
tasks. Unlike the decision tree, the random forest is not
based on the Euclidean distance. The random forest works
by first assuming the number of cases as K in the training
data, then it takes the k samples and uses this as the training
data for growing the tree. If there are p input variables,
Page 3
CMP7206 Data Mining

consider the number of variables as m, which is less than p,
so that each node, the variable selected, will be out of the p.
the best split of this m is used to split the node. Hence the
tree is growing to the extent as possible without the need for
pruning, finally, the new data is predicted by aggregating the
prediction of the target trees(Sudhakar & Reddy, 2014).The
figure 3 shows the representation of the Random Forest.
Figure 3. Random Forest Model
5.5. Evaluation Techniques.

5.5.1. CONFUSION MATRIX:

This report uses critical measures from the confusion metrics
to evaluate the model performance. The confusion
metrics classify the actual and predicted data set as True
positive(TP), True Negative(TN), False Positive(FP), False
Negative(FN), and True Positive is positive in both actual
and predicted data. True negative is negative in both actual
and predicted data. False positive is negative but predicted
as positive. False negative is positive but predicted as negative.
The key metrics calculated from the confusion metrics are
Accuracy, Precison, Recall, Error rate.The figure 4 shows
the confusion metrics.
Accuracy is the total number of correct predictions out of
the total number of predictions made for the data set. Accu-
Figure 4. Confusion Matrix
racy is sometimes unsuitable for data with imbalanced classification
problems. Accuracy= (TP+FN)/(TP+TN+FP+FN)
Accuracy is sometimes unsuitable for data with an imbalanced
classification problem. Precision and Recall measures
overcome this issue.
The precision defines the number of optimistic class predictions
that belong to the positive class. In other words,
how many positive predictions are accurate? Precision defines
as the number of true positives out of the true positives
and false positives. Precision=(TP/(TP+FP)) Recall
defines predicted actual positive cases from the total number
of positive predictions; Recall or sensitivity is the number
of correct positive predictions. The Recall defines
the total number of True positives out of the True positives
and false negatives. It explains how sensitive accurate
predictions are. The Recall is calculated as below,
Recall=(TP/(TP+FN)) The error rate is calculated as the
number of wrong predictions out of the total number of the
dataset. Error rate=(FP+FN)/(TP+TN+FN+FP) Specificity
is the number of the correct negatives predictions out of the
total number of negatives,0 is the best specificity, and 0 is
the worst. Specificity=TN/(TN+FP)

5.5.2. K FOLD CROSS VALIDATION:

The K fold cross validation is another evaluation technique
used in this report to evaluate the model’s performance on
the data set. K fold procedure divides the data set into K
folds of equal size. one of the folds is kept as test data, and
the rest is considered train data. The model fits into the train
data. The exact process is repeated k times using a different
set as a test and the rest as train data set. The average of k
times is calculated as the predicted accuracy.

11. Conclusions and Future Work

Bank direct marketing campaign effectively markets the
products and services where customers are contacted directly
via phone call. Data mining and classification and
predictive algorithms can help banks predict the customer
who would subscribe to the term deposit(Moro et al., 2011).
This report used data mining techniques such as Decision
Tree(DT), Logistic Regression(LR), Naive Bayes(NB), and
Random Forest(RF).
Experiments were conducted to build the model on the train
data and applied to the test data. The K fold cross validation
and confusion metrics are used to evaluate the model
performance with a high accuracy rate.
Overall the Decision tree showed a slightly better outcome
than the rest of the models. Logistic Regression and Random
forest also have shown promising results. The attributes
duration, Month, and balance are identified as the essential
attributes that influenced the outcome of the predictions. In
future work, we wish to include the clustering and association
technique to group the customers before applying
predictive analytic.

References

Amponsah, A. A. and Pabbi, K. A. Enhancing direct marketing
using data mining: A case of yaa asantewaa rural
bank ltd. in ghana. International Journal of Computer
Applications, 975:8887, 2016.

Chitra, K. and Subashini, B. Data mining techniques and its
applications in banking sector. International Journal of
Emerging Technology and Advanced Engineering, 3(8):
219–226, 2013.

Elsalamony, H. A. Bank direct marketing analysis of data
mining techniques. International Journal of Computer
Applications, 85(7):12–22, 2014.

Farooqi, R., Iqbal, N., et al. Effectiveness of data mining
in banking industry: An empirical study. International
Journal of Advanced Research in Computer Science, 8
(5):827–830, 2017.

Hamid, A. J. and Ahmed, T. M. Developing prediction
model of loan risk in banks using data mining. Machine
Learning and Applications: An International Journal
(MLAIJ), 3(1):1–9, 2016.

Jayasree, D. V. and Balan, S. A review on data mining in
banking sector. American Journal of Applied Sciences,
10:1160–1165, 10 2013. doi: 10.3844/ajassp.2013.1160.
1165.

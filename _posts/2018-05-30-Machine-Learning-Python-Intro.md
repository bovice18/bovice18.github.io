---
layout: post
title: Python Machine Learning Introduction
---

This is a walkthrough of a simple Python machine learning example using open source libraries

## Recurring Breast Cancer Screening ##

We are going to use machine learning to try to help screen women for breast cancer.  Specifically we will be using data from women who have previously had breast cancer to screen for a reccurence of the cancer.

### Data
Dataset we will be using for this example:
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data

In this set there are 201 cases of "no-recurrence-events" and 85 cases of "recurrence-events

*Fields:*
   1. Class: no-recurrence-events, recurrence-events
   2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
   3. menopause: lt40, ge40, premeno.
   4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
   5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
   6. node-caps: yes, no.
   7. deg-malig: 1, 2, 3.
   8. breast: left, right.
   9. breast-quad: left-up, left-low, right-up, right-low, central.
  10. irradiat: yes, no.


### Pre-reqs
`sudo pip install scipy` - package containing Matplotlib and Pandas

`sudo pip install numpy` - package for Python computing

`sudo pip install matplotlib` - 2D plotting

`sudo pip install pandas` - data structures and analysis

`sudo pip install sklearn` - Python machine learning tool


### Import Libraries and Data

In ipyton import the necessary libraries
{% highlight python %}
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
{% endhighlight %}

Import the sample data set into a pandas data frame.  Names are whatever you would like to name the columns
{% highlight python %}
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
dataset = pandas.read_csv(url, names=names)
{% endhighlight %}

Most of the columns contain categorical data rather than numeric.  Our python libraries do not like features that are categorical.
We need to convert the categorical columns to binary indicator columns also known as dummy columns.  This process is called [one hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) 
Luckily pandas has a built in method to do this:
{% highlight python %}
# Dont need to convert "Class" but we want to convert all other non numeric fields
dataset = pandas.get_dummies(dataset, columns=["age", 'menopause','tumor-size','inv-nodes','node-caps', 'breast','breast-quad','irradiat'])
{% endhighlight %}

### Data Analysis
Next we want to split our data into "training" and "validation" sets.  We will us a validation set of 20% of the total data.
When defining **X** and **Y**, we are saying that X should be all rows of columns 0-41 and Y should be all rows of only column 0.
{% highlight python %}
# Split-out validation dataset
array = dataset.values
X = array[:,0:41]
Y = array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
{% endhighlight %}

Next we will loop through 6 common machine learning algorithms to see which is the most accurate:
{% highlight python %}
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
{% endhighlight %}

Example output (could differ and ignore output about colinear variables)

> LR: 0.666601 (0.076388)

> LDA: 0.622727 (0.068611)

> KNN: 0.688142 (0.070920)

> CART: 0.613834 (0.075783)

> NB: 0.561462 (0.158986)

> SVM: 0.702569 (0.095458)


We can compare the results of each algo graphically
{% highlight python %}
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
{% endhighlight %}
![Algo Compare]({{ "/images/Algo-Compare.png" }})

Let's pick one of the more successful algos and test it on our validation dataset.  I am going to select the k-nearest neighbor classifier (KNN)
{% highlight python %}
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
{% endhighlight %}
Now we can analyze the **Y_validation** dataset vs. the predictions made by the k-nearest neighbord classifier
{% highlight python %}
print(accuracy_score(Y_validation, predictions))
0.758620689655
{% endhighlight %}
The below confusion matrix is of the form:

true negatives - {0,0} 

false negatives - {1,0}

true positives - {1,1} 

false positives - {0,1}
{% highlight python %}
print(confusion_matrix(Y_validation, predictions))
[[37  4]
 [10  7]]
{% endhighlight %}
{% highlight python %}
print(classification_report(Y_validation, predictions))

                      precision    recall  f1-score   support

no-recurrence-events       0.79      0.90      0.84        41
   recurrence-events       0.64      0.41      0.50        17

         avg / total       0.74      0.76      0.74        58
{% endhighlight %}

import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing as prep
from sklearn import metrics as metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn import svm as svm
from sklearn import neural_network as neural_net

from yellowbrick.regressor import AlphaSelection
import math as math
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('C:/mnist.csv')
data_values = pd.read_csv('C:/mnist.csv').values

labels = data_values[:, 0]
digits = data_values[:, 1:]

zero = data.loc[data['label'] == 0].values[:, 1:]
one = data.loc[data['label'] == 1].values[:, 1:]
two = data.loc[data['label'] == 2].values[:, 1:]
three = data.loc[data['label'] == 3].values[:, 1:]
four = data.loc[data['label'] == 4].values[:, 1:]
five = data.loc[data['label'] == 5].values[:, 1:]
six = data.loc[data['label'] == 6].values[:, 1:]
seven = data.loc[data['label'] == 7].values[:, 1:]
eight = data.loc[data['label'] == 8].values[:, 1:]
nine = data.loc[data['label'] == 9].values[:, 1:]

print(data.iloc[:, 1:].sum(axis=0))

ink = np.array([sum(row) for row in digits])
zero_digits = np.array([math.log(np.count_nonzero(row == 0)) for row in digits])

# Assignment 1
# SUM
print(ink)
# MEAN
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
print(ink_mean)

# STANDARD DEV
ink_std = [np.std(ink[labels == i]) for i in range(10)]
print(ink_std)

print(zero_digits)
zero_digits_mean = [np.mean(zero_digits[labels == i]) for i in range(10)]
zero_digits_std = [np.std(zero_digits[labels == i]) for i in range(10)]
print(zero_digits_mean)
print(zero_digits_std)

ink = prep.scale(ink).reshape(-1, 1)
zero_digits = prep.scale(ink).reshape(-1, 1)

x_ink = ink
x_ink_zero = pd.DataFrame(data=np.column_stack((ink,zero_digits)))
x_zero_digits = zero_digits

zero = np.array([sum(row) for row in zero])
zero = prep.scale(zero).reshape(-1, 1)
x_zero = zero

# <number>_extra => concat the ink feature with the zero count feature
zero_extra = np.array([np.count_nonzero(row == 0) for row in zero])
zero_extra = prep.scale(zero_extra).reshape(-1, 1)
x_zero_extra = pd.DataFrame(data=np.column_stack((zero,zero_extra)))

one = np.array([sum(row) for row in one])
one = prep.scale(one).reshape(-1, 1)
one_extra = np.array([np.count_nonzero(row == 0) for row in one])
one_extra = prep.scale(one_extra).reshape(-1, 1)
x_one_extra = pd.DataFrame(data=np.column_stack((one,one_extra)))

two = np.array([sum(row) for row in two])
two = prep.scale(two).reshape(-1, 1)
two_extra = np.array([np.count_nonzero(row == 0) for row in two])
two_extra = prep.scale(two_extra).reshape(-1, 1)
x_two_extra = pd.DataFrame(data=np.column_stack((two,two_extra)))

three = np.array([sum(row) for row in three])
three = prep.scale(three).reshape(-1, 1)
three_extra = np.array([np.count_nonzero(row == 0) for row in three])
three_extra = prep.scale(three_extra).reshape(-1, 1)
x_three_extra = pd.DataFrame(data=np.column_stack((three,three_extra)))

four = np.array([sum(row) for row in four])
four = prep.scale(four).reshape(-1, 1)
four_extra = np.array([np.count_nonzero(row == 0) for row in four])
four_extra = prep.scale(four_extra).reshape(-1, 1)
x_four_extra = pd.DataFrame(data=np.column_stack((four,four_extra)))

five = np.array([sum(row) for row in five])
five = prep.scale(five).reshape(-1, 1)
five_extra = np.array([np.count_nonzero(row == 0) for row in five])
five_extra = prep.scale(five_extra).reshape(-1, 1)
x_five_extra = pd.DataFrame(data=np.column_stack((five,five_extra)))

six = np.array([sum(row) for row in six])
six = prep.scale(six).reshape(-1, 1)
six_extra = np.array([np.count_nonzero(row == 0) for row in six])
six_extra = prep.scale(six_extra).reshape(-1, 1)
x_six_extra = pd.DataFrame(data=np.column_stack((six,six_extra)))

seven = np.array([sum(row) for row in seven])
seven = prep.scale(seven).reshape(-1, 1)
seven_extra = np.array([np.count_nonzero(row == 0) for row in seven])
seven_extra = prep.scale(seven_extra).reshape(-1, 1)
x_seven_extra = pd.DataFrame(data=np.column_stack((seven,seven_extra)))

eight = np.array([sum(row) for row in eight])
eight = prep.scale(eight).reshape(-1, 1)
eight_extra = np.array([np.count_nonzero(row == 0) for row in eight])
eight_extra = prep.scale(eight_extra).reshape(-1, 1)
x_eight_extra = pd.DataFrame(data=np.column_stack((eight,eight_extra)))

nine = np.array([sum(row) for row in nine])
nine = prep.scale(nine).reshape(-1, 1)
nine_extra = np.array([np.count_nonzero(row == 0) for row in nine])
nine_extra = prep.scale(nine_extra).reshape(-1, 1)
x_nine_extra = pd.DataFrame(data=np.column_stack((nine,nine_extra)))

lr = linear_model.LogisticRegression().fit(x_ink, labels)
lr_zero = linear_model.LogisticRegression().fit(x_zero_digits, labels)
lr_2 = linear_model.LogisticRegression().fit(x_ink_zero, labels)


print(metrics.accuracy_score(labels, lr.predict(x_ink)))
print(metrics.accuracy_score(labels, lr_zero.predict(x_zero_digits)))
print(metrics.accuracy_score(labels, lr_2.predict(x_ink_zero)))

# Assignment 2
print(lr.predict(x_zero))
print(lr.predict(one))
print(lr.predict(two))
print(lr.predict(three))
print(lr.predict(four))
print(lr.predict(five))
print(lr.predict(six))
print(lr.predict(seven))
print(lr.predict(eight))
print(lr.predict(nine))

# Assignment 3 + 4
print(lr_2.predict(x_zero_extra))
print(lr_2.predict(x_one_extra))
print(lr_2.predict(x_two_extra))
print(lr_2.predict(x_three_extra))
print(lr_2.predict(x_four_extra))
print(lr_2.predict(x_five_extra))
print(lr_2.predict(x_six_extra))
print(lr_2.predict(x_seven_extra))
print(lr_2.predict(x_eight_extra))
print(lr_2.predict(x_nine_extra))

# TODO: Check confusion matrices, which vars to use etc
print(metrics.confusion_matrix(labels, lr.predict(x_ink)))
#print(metrics.confusion_matrix(labels, lr_zero.predict(x_zero_digits)))
print(metrics.confusion_matrix(labels, lr_2.predict(x_ink_zero)))

# Assignment 5
data_copy = data
labels_copy = labels
sample_5000 = data.sample(5000).iloc[:, 1:]
sample_5000_labels = data.sample(5000).iloc[:, 0]

data_copy = pd.concat([data_copy, sample_5000, sample_5000]).drop_duplicates(keep=False)
labels_copy = data_copy.values[:, 0]
data_copy = data_copy.iloc[:, 1:]

sample_5000 = prep.scale(sample_5000)

# LASSO - find complexity parameters and estimated accuracy
print('LASSO')
lasso = linear_model.Lasso()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

grid = dict()
grid['alpha'] = np.arange(-2, 2, 0.1)
search = GridSearchCV(lasso, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


lasso_fit = search.fit(sample_5000, sample_5000_labels)
print(lasso_fit.alpha)
y_lasso = lasso_fit.predict(data_copy)
print(metrics.mean_absolute_error(labels_copy, y_lasso))




# Support vector - find complexity parameters and estimated accuracy
#supp_vec = svm.SVC().fit(sample_5000, sample_5000_labels)
#y_vec = supp_vec.predict(data_copy)
#print(metrics.mean_absolute_error(labels_copy, y_vec))

# Neural network - find complexity parameters and estimated accuracy
#neural_network = neural_net.MLPClassifier().fit(sample_5000, sample_5000_labels)
#y_neural = neural_network.predict(data_copy)
#print(metrics.mean_absolute_error(labels_copy, y_neural))




import os
import datetime as dt
import numpy as np

from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from util import load_data

################################################################################

# Classification data
target_names=['Female', 'Male']

dir_label = [
    ['badeer-r', 1], ['benson-r', 1], ['blair-l', 0],
    ['cash-m', 0], ['corman-s', 1], ['hain-m', 1]]

dataset = load_data(dir_label)

X = np.array(dataset[0])
y = dataset[1]

# Sci-Kit Learn Supprt Vector Machine Classifiers
# Train/Test split model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 20)

# SVC with Linear Kernel
lin = svm.SVC(kernel='linear', C = 1.0).fit(X_train, y_train)
y_lin_pred = lin.predict(X_test)
acc = metrics.accuracy_score(y_test, y_lin_pred)
cfm = metrics.confusion_matrix(y_test, y_lin_pred)
report = classification_report(y_test, y_lin_pred, target_names=target_names)
    
# SVC with radial basis kernel
rbf = svm.SVC(kernel='rbf', C = 1.0).fit(X_train, y_train)
y_rbf_pred = rbf.predict(X_test)
acc2 = metrics.accuracy_score(y_test, y_rbf_pred)
cfm2 = metrics.confusion_matrix(y_test, y_rbf_pred)
report2 = classification_report(y_test, y_rbf_pred, target_names=target_names)

# Write to file
filename = 'dataset/results.txt'
file_exists = os.path.exists(filename)
append_write = 'a' if(file_exists) else 'w'
f = open(filename, append_write)

# Write Data information
f.write("---------------------------------------------------------------------\n")
f.write("Date: " + str(dt.datetime.now().strftime("%m-%d-%Y")) + '\n')
f.write("Data Set: " + str(len(X)) + "\n")
f.write("X Tranining Set: " + str(len(X_train)) + " ; X Test Set: " + str(len(X_test)) + "\n")
f.write("Y Tranining Set: " + str(len(y_train)) + " ; Y Test Set: " + str(len(y_test)) + "\n")
f.write("\n")

# Write Linear metrics
f.write("Linear Accuracy: " + "{0:.4f}".format(acc) + "\n")
f.write("Confusioin Matrix: " + str(cfm.ravel()) + "\n")
f.write("Classification Report:\n" + report + "\n")

# Write Radial Basis metrics
f.write("Radial Basis Function Accuracy: " + "{0:.4f}".format(acc2) + "\n")
f.write("Confusioin Matrix: " + str(cfm2.ravel()) + "\n")
f.write("Classification Report:\n" + report2 + "\n")
f.write("\n")
f.close()

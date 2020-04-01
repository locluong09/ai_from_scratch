from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn import datasets


from ai.utils.data_preprocessing import min_max_scaler, normalize_scaler
from ai.utils.model_selection import train_test_split
from ai.utils.data_manipulation import accuracy_score

from ai.machine_learning import KNN




def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
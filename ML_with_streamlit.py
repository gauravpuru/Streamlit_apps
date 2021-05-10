import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
st.title("This is my first streamlit application")
st.write("""
# Explporing different datasets
## Which classifier is the best?
""")
dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine"))
classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

def get_dataset(dataset):
    if dataset == "Iris":
        data = datasets.load_iris()
    elif dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

x,y = get_dataset(dataset_name)

st.write("Shape of the selected dataset is ", x.shape)
st.write("Number of the classes in the dataset is ", len(np.unique(y)))

def add_parameter_ui(classifier):
    params = dict()
    if classifier == "KNN":
        k = st.sidebar.slider("K",1,15)
        params["K"] = k
    elif classifier == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        no_of_estimators = st.sidebar.slider("no. of estimators",1,100)
        params["max_depth"] = max_depth
        params["no_of_estimators"] = no_of_estimators
    return params
params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['no_of_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)
    return clf
clf = get_classifier(classifier_name,params=params)
# Time to do some classification

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)
clf.fit(xtrain,ytrain)
ypred = clf.predict(xtest)

acc = accuracy_score(ytest, ypred)
st.write(f"classifier = {clf}")
st.write(f"accuracy = {acc}")

#plotting the results
pca = PCA(2)
x_projected = pca.fit_transform(x)
x1 = x_projected[:,0]
x2 = x_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8, cmap='viridis')
plt.xlabel("PC_1")
plt.ylabel("PC_2")
plt.colorbar()
st.pyplot(fig)
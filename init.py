#%%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              VotingClassifier)

from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data(filename):
    data = pd.read_csv(filename,header=None,delim_whitespace=True)
    return data.to_numpy()

def load_datasets():
    X_test = load_data('./HAPT-Data/Test/X_test.txt')
    y_test = load_data('./HAPT-Data/Test/y_test.txt')
    X_train = load_data('./HAPT-Data/Train/X_train.txt')
    y_train = load_data('./HAPT-Data/Train/y_train.txt')

    #flatten y sets
    y_test =y_test.ravel()
    y_train = y_train.ravel()
    return X_test,y_test,X_train,y_train

X_test,y_test,X_train,y_train = load_datasets()
print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)

#%%
X_test[0]


def set_models():
    models = dict()
    models['Gaussian Naive Bayes Classifier'] = GaussianNB()
    #models['Nearest Neighbors Classifier'] = KNeighborsClassifier()
    models['Random Forest Classifier'] = RandomForestClassifier()
    #models['AdaBoost Classifier'] = AdaBoostClassifier()
    models['Decision Tree Classifer'] = DecisionTreeClassifier()
    models['Ridge Classifier'] = RidgeClassifier()
    models['Stochastic Gradient Descent Classifier'] = SGDClassifier()
    #models['Gaussian Process Classifier'] = GaussianProcessClassifier()
    return models

models = set_models()

def evaluate_model(model,X_train,y_train,X_test,y_test):
    #train model 
    start = time.time()
    model.fit(X_train,y_train)
    training_time = time.time() - start
    test_time = time.time() - start
    #predict
    predictions = model.predict(X_test)
    
    #sum errors
    errors = (y_test != predictions).sum()
    # calculate precision
    accuracy = 1.00 - (errors / X_test.shape[0])
    finish_time = time.time() - start

    print ( '%d errors on %d' % (errors, X_test.shape[0]))
    print( 'accuracy : %.2f %%' % (accuracy*100.00) )
    print(f'accuracy2.0 : {accuracy * 100.00}' )
    print( 'training time : %.10f s' % (training_time) )
    print(f'training time : {training_time}s')
    print(f'finish time : {finish_time} s')
    data = [accuracy*100.00, training_time, finish_time]
    return data

#%%
def evaluate():
    results = []
    results_dict = {}
    index = 0
    for key,model in models.items():
        print(f'{key}')
        evaluation = evaluate_model(model,X_train,y_train,X_test,y_test)
        model_results = [key]+ evaluation
        results_dict[key] = evaluation
        results.append(model_results)
    return results,results_dict
        
results,results_dict = evaluate()
print(results)
# %%
print(evaluate_model(models['Gaussian Naive Bayes Classifier'],X_train,y_train,X_test,y_test))

# %%
def sort(index):
    new_array = [i[0:index+1:index] for i in results]
    new_array = sorted(new_array,key=lambda x : x[1],reverse=True)
    return new_array
print(results)
print(sort(2))
# %%
accuracy_arr = sort(1)
training_time_arr = sort(2)
finish_time_arr = sort(3)

#%%
time_dict = {}
for i in range(0,len(results)):
    time_dict[results[i][0]] = results[i][2:]
print(time_dict)

# %%
df3 = pd.DataFrame.from_dict(time_dict,orient='index',)
df3.plot.bar()

# %%
acc01user01 = load_data('./HAPT-Data/RawData/acc_exp01_user01.txt')
df4 = pd.DataFrame(acc01user01,columns=['X','Y','Z'])
df4.plot()


# %%
#print(acc01user01)


%matplotlib inline
fig = plt.figure()
ax = plt.axes(projection='3d')
x_axis = [ acc01user01[i][0] for i in range(len(acc01user01))]
y_axis = [ acc01user01[i][1] for i in range(len(acc01user01))]
z_axis = [ acc01user01[i][2] for i in range(len(acc01user01))]
ax.scatter3D(x_axis, y_axis, z_axis)
# %%
print(len(acc01user01))

#%%
if not os.path.exists('Models'):
    os.mkdir('Models')
for name,model in models.items():
    foldername = 'Models/'+ name.lower().replace(' ','_')
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    joblib.dump(model, filename = foldername+'_model.sav')




# %%

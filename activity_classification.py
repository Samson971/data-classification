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
import matplotlib.patches as mpatches

"""
Loading the data
"""
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

"""
data modelization
"""
#%%
activity_color_map = {
    1: ['WALKING','lightblue'],
    2: ['WALKING_UPSTAIRS','yellow'],
    3: ['WALKING_DOWNSTAIRS','lightgreen'],
    4: ['SITTING','red'],
    5: ['STANDING','orangered'],
    6: ['LAYING','orange'],
    7: ['STAND_TO_SIT','black'],
    8: ['SIT_TO_STAND','sienna'],
    9: ['SIT_TO_LIE','darkblue'],
    10:[ 'LIE_TO_SIT','darkgreen'],
    11:[ 'STAND_TO_LIE','magenta'],
    12:[ 'LIE_TO_STAND','grey'],
}
color_legend = [mpatches.Patch(color = value[1], label = value[0]) for (key,value) in activity_color_map.items()]


def to_color(activity_id):
    return activity_color_map[activity_id][1]


"""Mean of body-Acc for XYZ"""
plot_data = [[X_train[i][0],X_train[i][1],X_train[i][2],y_train[i]] for i in range(len(X_train))]
fig = plt.figure()
ax = plt.axes(projection='3d')
x_axis = [ X_train[i][0] for i in range(len(X_train))]
y_axis = [ X_train[i][1] for i in range(len(X_train))]
z_axis = [ X_train[i][2] for i in range(len(X_train))]
color = [ to_color(y_train[i]) for i in range(len(y_train))]
ax.scatter3D(x_axis, y_axis, z_axis,color=color)
lgd = ax.legend(handles=color_legend,bbox_to_anchor=(-0.1,1))
ax.set_title('Mean of body-Acc for XYZ')

#%%
"""Standard Deviation of body-Acc for XYZ"""
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
x_axis2 = [ X_train[i][3] for i in range(len(X_train))]
y_axis2 = [ X_train[i][4] for i in range(len(X_train))]
z_axis2 = [ X_train[i][5] for i in range(len(X_train))]
color = [ to_color(y_train[i]) for i in range(len(y_train))]
ax2.scatter3D(x_axis2, y_axis2, z_axis2,color=color)
ax2.legend(handles=color_legend,bbox_to_anchor=(-0.1,1))
ax2.set_title('Standard Deviation of body-Acc for XYZ')

#%%
"""Median Deviation of body-Acc for XYZ"""
fig3 = plt.figure()
ax3 = plt.axes(projection='3d')
x_axis3 = [ X_train[i][6] for i in range(len(X_train))]
y_axis3 = [ X_train[i][7] for i in range(len(X_train))]
z_axis3 = [ X_train[i][8] for i in range(len(X_train))]
color = [ to_color(y_train[i]) for i in range(len(y_train))]
ax3.scatter3D(x_axis3, y_axis3, z_axis3,color=color)
ax3.legend(handles=color_legend,bbox_to_anchor=(-0.1,1))
ax3.set_title('Median Deviation of body-Acc for XYZWe train the models and test them to and gather data such as the accuracy​')

#%%
"""Energy of body-Acc for XYZ"""
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')
x_axis4 = [ X_train[i][16] for i in range(len(X_train))]
y_axis4 = [ X_train[i][17] for i in range(len(X_train))]
z_axis4 = [ X_train[i][18] for i in range(len(X_train))]
color = [ to_color(y_train[i]) for i in range(len(y_train))]
ax4.scatter3D(x_axis4, y_axis4, z_axis4,color=color)
ax4.legend(handles=color_legend,bbox_to_anchor=(-0.1,1))
ax4.set_title('Energy of body-Acc for XYZ')

#%%
"""Correlation of body-Acc for XYZ"""
fig5 = plt.figure()
ax5 = plt.axes(projection='3d')
x_axis5 = [ X_train[i][37] for i in range(len(X_train))]
y_axis5 = [ X_train[i][38] for i in range(len(X_train))]
z_axis5 = [ X_train[i][39] for i in range(len(X_train))]
color = [ to_color(y_train[i]) for i in range(len(y_train))]
ax5.scatter3D(x_axis5, y_axis5, z_axis5,color=color)
ax5.legend(handles=color_legend,bbox_to_anchor=(-0.1,1))
ax5.set_title('Correlation of body-Acc for XYZ')

#%%
"""Mean of body-Gyro for XYZ"""
fig6 = plt.figure()
ax6 = plt.axes(projection='3d')
x_axis6 = [ X_train[i][120] for i in range(len(X_train))]
y_axis6 = [ X_train[i][121] for i in range(len(X_train))]
z_axis6 = [ X_train[i][122] for i in range(len(X_train))]
color = [ to_color(y_train[i]) for i in range(len(y_train))]
ax6.scatter3D(x_axis6, y_axis6, z_axis6,color=color)
ax6.legend(handles=color_legend,bbox_to_anchor=(-0.1,1))
ax6.set_title('Mean of body-Gyro for XYZ')


"""
creatting the models
training them and get the results
"""
#%%
def set_models():
    models = dict()
    models['gaussian_naive_bayes_classifier'] = GaussianNB()
    models['nearest_neighbors_classifier'] = KNeighborsClassifier()
    models['random_forest_classifier'] = RandomForestClassifier()
    models['decision_tree_classifer'] = DecisionTreeClassifier()
    models['ridge_classifier'] = RidgeClassifier()
    models['stochastic_gradient_descent_classifier'] = SGDClassifier()
    #(too long) models['gaussian_process_classifier'] = GaussianProcessClassifier()
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
###results are in form key,accurracy,training time, total time
#%%
#save the model in the api project so we can use them
if not os.path.exists('api_project/models'):
    os.mkdir('api_project/models')
for name,model in models.items():
    filename = 'api_project/models/'+ name.lower().replace(' ','_')+'_model.sav'
    if not os.path.exists(filename):
        joblib.dump(model, filename = filename)



# %%
print(evaluate_model(models['gaussian_process_classifier'],X_train,y_train,X_test,y_test))

# %%
def sort(index):
    new_array = [i[0:index+1:index] for i in results]
    new_array = sorted(new_array,key=lambda x : x[1],reverse=True)
    return new_array
print(results)
print(sort(2))
# %%
## more accurate
print(sort(1))

##fastest
#training
print(sort(2))
#total
print(sort(3))



"""
plotting the results
"""
#%%

time_dict = {key:{'training time':value[1],'total time':value[2]} for (key,value) in results_dict.items()}
df3 = pd.DataFrame.from_dict(time_dict,orient='index')
ax = df3.plot.bar()
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points',rotation=45)

#%%
print (results_dict)
accuracy = {key:value[0] for (key,value) in results_dict.items()}
df6 = pd.DataFrame.from_dict({'accuracy':accuracy})
ax2 = df6.plot.bar()
for p in ax2.patches: 
    ax2.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')



"""
predict some values from the api
must run the server first

model must be one of these strings:
    gaussian_naive_bayes_classifier
    nearest_neighbors_classifier
    random_forest_classifier
    decision_tree_classifer
    ridge_classifier
    stochastic_gradient_descent_classifier
"""
#%%
def predict(data,model):
    endpoint = 'http://localhost:8000/predict/'
    param = {'data':data,'model':model}
    r = requests.post(url = endpoint, json = param)
    return dict(json.loads(r.content.decode('utf-8')))


#data must be a 2 dimensionnal array converted in a list
#(reshape) if it contains one array
#%%
predict(X_test[1:3].tolist(),'gaussian_naive_bayes_classifier')
#%%
predict(X_test[1].reshape(1,-1).tolist(),'gaussian_naive_bayes_classifier')
predict(X_test[1:123].tolist(),'gaussian_naive_bayes_classifier')

# %%

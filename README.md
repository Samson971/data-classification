# Activity Classification

## Dataset

The dataset is extract from an experiment where the activity of 30 people was measured.


## Data Visualization

We gathered data from 3 axis (XYZ) features to create scatter plots.
The features used are :

- tBodyAcc-Mean-XYZ
- tBodyAcc-STD-XYZ
- tBodyAcc-Mad-XYZ
- tBodyAcc-Energy-XYZ
- tBodyAcc-Correlation-XYZ
- tBodyGyro-Mean-XYZ

In each plot, we can observe a pattern, points corresponding to the same activities tend to gather in the same areas.

## Models Training



## Results


## API
Creation of an Api with one endpoint : ```predict/```

Its a Post Method and requires 2 argument in the body of the request: data and model

- data: the array of values from an activity (561,n).\
    It must be multidimensionnal but also must be passed as a **list**.\
    **If its a one dimension np array, it must be reshaped with (1,-1)**

- model: a string to choose the model for the prediction
    Only those values are allowed : 
    ```gaussian_naive_bayes_classifier
    nearest_neighbors_classifier
    random_forest_classifier
    decision_tree_classifer
    ridge_classifier
    stochastic_gradient_descent_classifier
    ```

The response is a json document formed as :

```json
{"results": [
    {"id": "activity id of the first array", "label": "label of the first array"},
    {"id": "activity id of the second array", "label": "label of the second array"},
    {"id": "activity id of the third array", "label": "label of the third array"},
]}
```

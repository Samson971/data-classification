#%%
import pandas as pd
import requests

def load_data(filename):
    data = pd.read_csv(filename,header=None,delim_whitespace=True)
    return data.to_numpy()

X_test = load_data('./HAPT-Data/Test/X_test.txt')

# %%
print(len(X_test[0]))

# %%
endpoint = 'http://localhost:8000/predict/'
param = {'data':X_test[0].reshape(1,-1).tolist()}
r = requests.post(url = endpoint, json = param)
print(r.content)
# %%
import requests
print (requests.__file__)

# %%

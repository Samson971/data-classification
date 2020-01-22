#%%
import pandas as pd
import requests
import json

def load_data(filename):
    data = pd.read_csv(filename,header=None,delim_whitespace=True)
    return data.to_numpy()

X_test = load_data('./HAPT-Data/Test/X_test.txt')

# %%
print(len(X_test[0]))

# %%
endpoint = 'http://localhost:8000/predict/'
param = {'data':X_test[0:2].tolist()}
r = requests.post(url = endpoint, json = param)
print(r.content.decode('utf-8'))
print(r.content)
print (r.request.body)
# %%
import requests
print (requests.__file__)

# %%
print((X_test[0:12]).shape)

# %%
testdict = {
    'test':['data',0,12,'test']
}
testjson = json.dumps(testdict)
print(testjson)
load = json.loads(testjson)
print(load)
# %%

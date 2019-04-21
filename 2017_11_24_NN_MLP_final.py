
# coding: utf-8

# In[96]:

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import time as time
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[97]:

data = pd.read_csv('/Users/zwz/Documents/zwz/rutgers/2017 Fall/data mining/project/train.csv', index_col= 0)
#df_test = pd.read_csv('test.csv', index_col= 0)


# In[98]:

df_train, df_test = train_test_split(data, test_size=0.2, random_state=7)


# In[43]:

df_train.columns


# In[44]:

selected_features = ['bathrooms', 'bedrooms', 'latitude',
       'longitude', 'price', 'price_per_bedrm',
       'price_per_bathrm', 'manager_id_categorized', 'building_id_categorized',
       'street_add_categorized', 'display_add_categorized', 'elevator',
       'cats_allowed', 'dogs_allowed', 'hardwood_floors', 'doorman',
       'dishwasher', 'laundry', 'no_fee', 'fitness_center', 'pre_war',
       'roof_deck', 'outdoor_space', 'dining_room', 'high_speed_internet',
       'balcony', 'terrace', 'swimming_pool', 'new_construction', 'exclusive',
       'loft', 'wheelchair_access', 'simplex', 'fireplace', 'lowrise',
       'garage', 'reduced_fee', 'furnished', 'multi_level', 'high_ceilings',
       'super', 'parking', 'renovated', 'green_building', 'storage',
       'stainless_steel_appliances', 'concierge', 'light', 'exposed_brick',
       'eat_in_kitchen', 'granite_kitchen', 'bike_room', 'walk_in_closet',
       'marble_bath', 'valet', 'subway', 'lounge', 'short_term_allowed',
       'children_playroom', 'no_pets', 'central_air', 'luxury_building',
       'view', 'virtual_doorman', 'courtyard', 'microwave', 'sauna',
       'len_of_features', 'bedr_bathr', 'month', 'hour', 'Friday', 'Monday',
       'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
       'len_of_description', 'upper_percent_of_des', 'num_of_symbols',
       'email_dummy', 'phone_dummy', 'same_address', 'straight_distance',
       'location_label']
x_train = df_train[selected_features].values
x_test = df_test[selected_features].values
y_train = df_train['interest_level']
y_test = df_test['interest_level']
print(x_train.shape, x_test.shape)


# In[45]:

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[72]:

# Classifier
clf_nn = MLPClassifier(solver='lbfgs', random_state=1)
params = {
    'alpha': [1e-6,1e-5, 1e-4],
    'activation': ['relu', 'tanh'],
    'hidden_layer_sizes': [(30, 30, 5), (20, 20, 20), (30, 30, 5), (10, 30, 5), (10, 40, 5)]
}
gs_nn = GridSearchCV(clf_nn, param_grid=params, scoring='neg_log_loss', n_jobs=2, cv=5, verbose=2, refit=True)
start = time.time()
gs_nn.fit(x_train,y_train)
print('- Time: %.2f minutes' % ((time.time() - start)/60))
print('- Best score: %.4f' % gs_nn.best_score_)
print('- Best params: %s' % gs_nn.best_params_)


# ### Best NN model

# In[89]:

mlp_best = MLPClassifier(activation='relu', alpha=1e-06, hidden_layer_sizes=(10, 40, 5), random_state=7)


# In[92]:

mlp_best.fit(x_train, y_train)


# In[93]:

mlp_preds_d = mlp_best.predict(x_test)
mlp_preds_p = mlp_best.predict_proba(x_test)


# In[95]:

print('log loss:', log_loss(y_test, mlp_preds_p))
print(classification_report(y_test, mlp_preds_d, target_names=['low','mid','high']))
print(confusion_matrix(y_test, mlp_preds_d))


# In[100]:




# In[ ]:




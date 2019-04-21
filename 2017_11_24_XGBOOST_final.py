
# coding: utf-8

# In[57]:

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:

data = pd.read_csv('train.csv')


# In[3]:

train_df, test_df = train_test_split(data, test_size=0.3, random_state=7)


# In[4]:

features_to_use=['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'price_per_bedrm',
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


# In[5]:

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# In[6]:

selected_features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'price_per_bedrm',
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

for i, f in enumerate(features_to_use):
    print(i, f, "Used" if f in selected_features else '**Not Used**')


# In[55]:

train_X = train_df[selected_features].values
test_X = test_df[selected_features].values

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = train_df['interest_level'].values
test_y = test_df['interest_level'].values
print(train_X.shape, test_X.shape)


# In[58]:

scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)


# In[8]:

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break


# In[59]:

preds, model = runXGB(train_X, train_y, test_X, num_rounds=900)


# In[ ]:

out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
#out_df.to_csv("result.csv", index=False)


# In[60]:

xgb_preds_d = []
xgb_preds_p = preds
for x in preds:
    index = 0
    for prob in x:
        max_value = max(x)
        if prob == max_value:
            xgb_preds_d.append(index)
            index = 0
        else:
            index += 1
#xgb_preds_d = np.array(xgb_preds_d).reshape((-1,3))


# In[54]:

print('log loss:', log_loss(test_y, xgb_preds_p))
print(classification_report(test_y, xgb_preds_d, target_names=['low','mid','high']))
print(confusion_matrix(test_y, xgb_preds_d))


# In[61]:

print('log loss:', log_loss(test_y, xgb_preds_p))
print(classification_report(test_y, xgb_preds_d, target_names=['low','mid','high']))
print(confusion_matrix(test_y, xgb_preds_d))


# In[53]:

get_ipython().magic('matplotlib notebook')
for i, f in enumerate(selected_features):
    print(i, f)
xgb.plot_importance(model, max_num_features=20)
plt.show()


# In[ ]:




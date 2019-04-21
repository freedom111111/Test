#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:50:30 2017

@author: zwz
"""

# In[1]:
import pandas as pd
import numpy as np
import time as time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline

# In[2]


df=pd.read_csv('/Users/zwz/Documents/zwz/rutgers/2017 Fall/data mining/project/train.csv',index_col=0)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=7)
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


# In[3]:
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[4]:
clf_rf = RandomForestClassifier(criterion='gini',random_state=7)
params ={'n_estimators':[100,1000],'max_depth': [5,None],'max_features': [8,10,'sqrt','log2']}
gs_rf = GridSearchCV(clf_rf, param_grid=params, verbose=3,n_jobs=2,scoring='neg_log_loss',cv=5,refit=True)
start = time.time()
gs_rf.fit(x_train,y_train)
print('- Time: %.2f minutes' % ((time.time() - start)/60))
print('- Best score: %.4f' % gs_rf.best_score_)
print('- Best params: %s' % gs_rf.best_params_)

# In[5]:
rf_best=RandomForestClassifier(criterion='gini',n_estimators=1000,max_depth=None,max_features=10,random_state=7)

# In[6]:
rf_best.fit(x_train,y_train)

rf_pred_test = rf_best.predict(x_test)
rf_pred_prob_test = rf_best.predict_proba(x_test)
rf_pred_train = rf_best.predict(x_train)
rf_pred_prob_train = rf_best.predict_proba(x_train)

# In[7]
print('log loss:', log_loss(y_test, rf_pred_prob_test))
print(classification_report(y_test, rf_pred_test, target_names=['low','mid','high']))
print(confusion_matrix(y_test, rf_pred_test))

print('log loss:', log_loss(y_train, rf_pred_prob_train))
print(classification_report(y_train, rf_pred_train, target_names=['low','mid','high']))
print(confusion_matrix(y_train, rf_pred_train))
# In[8]
out_train_df = pd.DataFrame(rf_pred_prob_train)
out_train_df.columns = ['high', 'medium', 'low']
out_train_df['listing_id'] = df_train.listing_id.values
out_train_df.to_csv('/Users/zwz/Documents/zwz/rutgers/2017 Fall/data mining/project/RF_train_result.csv', index=False)


out_test_df = pd.DataFrame(rf_pred_prob_test)
out_test_df.columns = ['high', 'medium', 'low']
out_test_df['listing_id'] = df_test.listing_id.values
out_test_df.to_csv('/Users/zwz/Documents/zwz/rutgers/2017 Fall/data mining/project/RF_test_result.csv', index=False)



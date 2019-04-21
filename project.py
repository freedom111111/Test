#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:33:38 2017

@author: zwz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn as sk

## original data
traindf=pd.read_json('/Users/zwz/Documents/zwz/rutgers/2017 Fall/data mining/project/train.json')
## get y and categorize it
ytrain=traindf['interest_level']
ytrain=ytrain.astype('category')
ytrain=ytrain.cat.reorder_categories(['low', 'medium', 'high'], ordered=True)
ytrain=ytrain.cat.codes
traindf['interest_level']=ytrain

## deal with price
traindf['price']=np.log(traindf['price'])
traindf['price_per_bedrm']=traindf.apply(lambda r: np.log(1+r['price']/r['bedrooms']) if r['bedrooms'] !=0 else 0,axis=1)
traindf['price_per_bathrm']=traindf.apply(lambda r: np.log(1+r['price']/r['bathrooms']) if r['bathrooms'] !=0 else 0,axis=1)


## manager id
managers=traindf['manager_id']
plt.hist(managers.value_counts()[10:],bins =100)
plt.yscale('log')
managers=pd.DataFrame(managers)
managers.columns=['manager_id']
class CategoricalFilter(object):
    def __init__(self, top_categories = 999):
        self.top_categories = top_categories
    
    def fit(self, series):
        counts = series.value_counts()
        self.category_mapper = dict(zip(counts.index[:self.top_categories],
                                    range(1, self.top_categories + 1)))
    def transform(self, series):
        return series.apply(lambda key: self.category_mapper.get(key, 0))
catfilter = CategoricalFilter()
catfilter.fit(managers['manager_id'])
manager_transformed = catfilter.transform(managers['manager_id'])
manager_transformed.head(20)
traindf['manager_id_categorized']=manager_transformed    

##  building id
traindf['building_id'].value_counts()
catfilter = CategoricalFilter()
catfilter.fit(traindf['building_id'])
building_transformed = catfilter.transform(traindf['building_id'])
building_transformed.head(20)
traindf['building_id_categorized']=building_transformed 

## street address
traindf['street_address'].value_counts()
catfilter = CategoricalFilter()
catfilter.fit(traindf['street_address'])
st_add_transformed = catfilter.transform(traindf['street_address'])
st_add_transformed.head(20)
traindf['street_add_categorized']=st_add_transformed 

## display address
traindf['display_address'].value_counts()
catfilter = CategoricalFilter()
catfilter.fit(traindf['display_address'])
display_st_transformed = catfilter.transform(traindf['display_address'])
display_st_transformed.head(20)
traindf['display_add_categorized']=display_st_transformed 


## features
import re
from six import string_types
FEATURES_MAP = {'elevator': 'elevator',
                'cats_allowed': r'(?<!\w)cats?(?!\w)|(?<!\w)(?<!no )pets?(?!\w)',
                'dogs_allowed': r'(?<!\w)dogs?(?!\w)|(?<!\w)(?<!no )pets?(?!\w)(?!: cats only)',
                'hardwood_floors': 'hardwood',
                'doorman': r'(?<!virtual )doorman',
                'dishwasher': 'dishwasher|dw(?!\w)',
                'laundry': r'laundry(?! is on the blo)',
                'no_fee': 'no fee',
                'fitness_center': r'fitness(?! goals)|gym',
                'pre_war': r'pre\s?war',
                'roof_deck': 'roof',
                'outdoor_space': 'outdoor|garden|patio',
                'dining_room': 'dining',
                'high_speed_internet': r'high.*internet',
                'balcony': r'balcon(y|ies)|private.*terrace',
                'terrace': 'terrace',
                'swimming_pool': r'pool(?! table)',
                'new_construction': 'new construction',
                'exclusive': r'exclusive( rental)?$',
                'loft': r'(?<!sleep )loft(?! bed)',
                'wheelchair_access': 'wheelchair',
                'simplex': 'simplex',
                'fireplace': ['fireplace(?! storage)', 'deco'], # looks for first regex, excluding matches of the second regex
                'lowrise': r'low\s?rise',
                'garage': r'garage|indoor parking',
                'reduced_fee': r'(reduced|low) fee',
                'furnished': ['(?<!un)furni', 'deck|inquire|terrace'],
                'multi_level': r'multi\s?level|duplex',
                'high_ceilings': r'(hig?h|tall) .*ceiling',
                'super': r'(live|site).*super',
                'parking': r'(?<!street )(?<!side )parking(?! available nearby)',
                'renovated': 'renovated',
                'green_building': 'green building',
                'storage': 'storage',
                'stainless_steel_appliances': r'stainless.*(appliance|refrigerator)',
                'concierge': 'concierge',
                'light': r'(?<!\w)(sun)?light(?!\w)',
                'exposed_brick': 'exposed brick',
                'eat_in_kitchen': r'eat.*kitchen',
                'granite_kitchen': 'granite kitchen',
                'bike_room': r'(?<!citi)(?<!citi )bike',
                'walk_in_closet': r'walk.*closet',
                'marble_bath': r'marble.*bath',
                'valet': 'valet',
                'subway': r'subway|trains?(?!\w)',
                'lounge': 'lounge',
                'short_term_allowed': 'short term',
                'children_playroom': r'(child|kid).*room',
                'no_pets': 'no pets',
                'central_air': r'central a|ac central',
                'luxury_building': 'luxur',
                'view': r'(?<!\w)views?(?!\w)|skyline',
                'virtual_doorman': 'virtual d',
                'courtyard': 'courtyard',
                'microwave': 'microwave|mw',
                'sauna': 'sauna'}

def _subparser(x):
    x = x.lower().replace('-', ' ').strip()
    if x[0] == '{':
        return [y.replace('"', '').strip() for y in re.findall(r'(?<=\d\s=\s)([^;]+);', x)]
    x = x.split(u'\u2022')
    return [z for y in x for z in re.split(r'[\.\s!;]!*\s+|\s+-\s+|\s*\*\s*', y)]

def _parser(x):
    return [z for z in [y.strip() for y in _subparser(x)] if len(z) > 0]

def _extract_features(features, feature_parser = lambda x: [x.lower()]):
	return [feature for ft in features for feature in feature_parser(ft)]

def _search_regex(regexes):
    if isinstance(regexes, string_types):
        filter_fun = lambda x: re.search(regexes, x) is not None
    else:
        filter_fun = lambda x: re.search(regexes[0], x) is not None and re.search(regexes[1], x) is None
    return lambda x: 1 if np.any([filter_fun(ft) for ft in x]) else 0

def get_dummies_from_features(series, dtype = np.float32):
    series = series.apply(lambda x: _extract_features(x, _parser))
    dummies = np.zeros((len(series), len(FEATURES_MAP)), dtype = dtype)
    for i, key in enumerate(FEATURES_MAP):
        dummies[:, i] = series.apply(_search_regex(FEATURES_MAP[key]))
    return dummies

feature_dummies=pd.DataFrame(get_dummies_from_features(traindf['features']))     
colnames=list(FEATURES_MAP.keys())
feature_dummies.columns=colnames
index=traindf.index
feature_dummies.index=index
traindf=pd.concat([traindf,feature_dummies],axis=1)     
lenlist=[]
for x in traindf['features']:
    lenlist.append(len(x))
traindf['len_of_features']=lenlist

## bedr+bathr
traindf['bedr_bathr']=traindf['bathrooms']+traindf['bedrooms']

## month day hour
traindf['created']= pd.to_datetime(traindf['created'], format='%Y-%m-%d %H:%M:%S')
traindf['month'] = traindf.created.dt.month
traindf['day_of_week'] = traindf.created.dt.weekday_name
traindf['hour'] = traindf.created.dt.hour
       
day_dummies = pd.get_dummies(traindf['day_of_week'])       
traindf=pd.concat([traindf,day_dummies],axis=1)       

## description
#1 lenth
lenlist=[]
for x in traindf['description']:
    lenlist.append(len(x.split()))
traindf['len_of_description']=lenlist

#2 uppercase words percentage
uppercent=[]
for x in traindf['description']:
    for y in x.split():
        y=''.join(i for i in y if i.isalpha())
        count=len([word for word in y if word.isupper()])
    if len(x.split())==0:
        uppercent.append(0)
    else:
        uppercent.append(count/len(x.split()))
len(uppercent)        
traindf['upper_percent_of_des']=uppercent

#3 number of special characters
symbols=['!','>','<','$','#','~','*']
symlist=[]       
for x in traindf['description']:
    count=len([r for r in x if r in symbols])
    symlist.append(count)
len(symlist)
traindf['num_of_symbols']=symlist


#4 email
traindf['email_dummy']=traindf.apply(lambda r:1 if "@" in r['description'] and '.com' in r['description'] else 0,axis=1)

#5 phone
phone_dummy=[]
for x in traindf['description']:
    y=re.findall('\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}',x)
    if y==[]:
        phone_dummy.append(0)
    else:
        phone_dummy.append(1)
traindf['phone_dummy']=phone_dummy

#6 is street add same as display address
traindf['same_address']=traindf.apply(lambda r: 1 if r['display_address'] in r['street_address'] else 0, axis=1)
traindf.tail()


## location longitude and latitude
x=traindf['latitude']
y=traindf['longitude']
data=pd.concat([x,y],axis=1)
plt.scatter(x,y)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0).fit(data) # 6 centers
labels=kmeans.labels_
centers=list(kmeans.cluster_centers_)

# plot kmeans   
X=x.reset_index(drop=True)
Y=y.reset_index(drop=True)  
 
plt.scatter(x,y,c=kmeans.labels_,marker='o')
plt.xlabel("latitude")  
plt.ylabel("longitude")  
plt.show()
data[(data['latitude']==0)&(data['longitude']==0)]

## distance
distance=[]
for i in range(len(data)):
    if labels[i]==0:
        x1center=centers[0][0]
        y1center=centers[0][1]
        temp=np.sqrt((X[i]-x1center)**2+(Y[i]-y1center)**2)
    elif labels[i]==1:
        x2center=centers[1][0]
        y2center=centers[1][1]
        temp=np.sqrt((X[i]-x2center)**2+(Y[i]-y2center)**2)
    elif labels[i]==2:
        x3center=centers[2][0]
        y3center=centers[2][1]
        temp=np.sqrt((X[i]-x3center)**2+(Y[i]-y3center)**2)
    elif labels[i]==3:
        x4center=centers[3][0]
        y4center=centers[3][1]
        temp=np.sqrt((X[i]-x4center)**2+(Y[i]-y4center)**2)
    elif labels[i]==4:
        x5center=centers[4][0]
        y5center=centers[4][1]
        temp=np.sqrt((X[i]-x5center)**2+(Y[i]-y5center)**2)
    else:
        x6center=centers[5][0]
        y6center=centers[5][1]
        temp=np.sqrt((X[i]-x6center)**2+(Y[i]-y6center)**2)
    distance.append(temp)
traindf['straight_distance']=distance
traindf['location_label']=list(labels)
## drop columns
drop_columns=['building_id','created','description','display_address','features','listing_id','street_address','manager_id','photos','day_of_week']
traindf=traindf.drop(drop_columns,axis=1)
traindf.tail()
traindf.columns
traindf.to_csv(r'/Users/zwz/Documents/zwz/rutgers/2017 Fall/data mining/project/train.csv')




from flask import Flask, Response, request, render_template, jsonify, abort
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import joblib
from newspaper import fulltext
import requests
import sqlite3
from sqlite3 import Error
from scipy import stats
import os
import psycopg2
from bs4 import BeautifulSoup
##########
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#########
# this function parses the raw html to extract data to use a input into the ML model:
def parse_features_from_html(soup):
    # extracting text:
    visible_text = []
    texts = soup.body.findAll(text=True)
    for i in texts:
        if len(i) > 2:
            if not "mso" in i:
                if not "endif" in i:
                    if not "if !vml" in i:
                        visible_text.append(i)

    # extracting links:
    links = soup.find_all('a')
    # extracting images:
    img_tags = soup.find_all('img')

    title_count = len(visible_text[0].split())
    word_count = len(' '.join(visible_text).split())
    link_count = len(links)
    img_count = len(img_tags)

    width_list = [i['width'] for i in img_tags]
    max_idx = width_list.index(max(width_list))
    main_image = img_tags[max_idx]
            
    return visible_text,links,img_tags,title_count,word_count,link_count,img_count,main_image
########

# this function subsets the data based on the user inputs and then analyzed the probability outcomes of altenate inputs
def make_predictions(target_variable, 
                        campaign_type, 
                        industry_type, 
                        link_count,
                        img_count, 
                        title_count,
                        word_count): 

    data = pd.read_csv('data/wrangled_data.csv', index_col=0)
    data_copy = data.copy()

    selected_dependent_target_variable =  target_variable # label
    selected_campaign =  campaign_type # no_opener
    selected_industry = industry_type # automotive

    # need at least target variables in droppin colums below:
    target_variables = ['label', 'unsubscribed', 'open_rate', 'click_through', 'abandoned_cart']
    campaign_types = ['Promotional', 'Survey', 'No_Opener', 'Revenue_Based', 'Abandoned_Cart', 'Engagement_Campaign']
    industry_types = ['Medical', 'Hospitality', 'Industrial', 'Automotive', 'Real_Estate']

    # filtering for campaign and industry
    data = data[(data.campaign_type == selected_campaign) & (data.industry == selected_industry)]

    # dropping all other dependent variables not being used in analysis as we don't have this data for the test data

    # drop 'text' and 'title' for now as not necessarily performing NLP on the data at this type 
    # NLP and Computer Vision change scope and scale of the project
    X = data.drop(columns=target_variables)
    X = X.drop(columns=['text','title'])

    # Dropping length to just make prediction based on bin, not specific value, 
    # otherwise I would need to separate models, with and without the specific value

    # might have to do this for video_length/title_length if going to switch out the bin values 
    X = X.drop(columns=['length'])

    y = data[selected_dependent_target_variable]

    ############
    X_temp = X.copy()
    X = pd.get_dummies(X)

    #############

    # creating train test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    ###########

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    ####
    # single_input = X_temp.iloc[prediction_row_index]
    # single_input = pd.DataFrame(single_input).T

    # setting values for single input from user data:
    # (note: need to eventually change all inputs with data extracted with BS4)
    single_input = X_temp.iloc[0].copy()
    single_input = pd.DataFrame(single_input).T
    single_input.values[:] = 0
    # single_input['length'] = word_count
    single_input['num_pics'] = img_count
    # this is not correct needs to change! (original dataframe does not have num links, need to add)
    # single_input['num_videos'] = link_count


########
    if title_count < 4:
        single_input['title_length_binned'] = 'small'
    elif title_count < 8:
        single_input['title_length_binned'] = 'med'        
    elif title_count < 12:
        single_input['title_length_binned'] = 'big'               
    elif title_count < 16:
        single_input['title_length_binned'] = 'bigger'       
    else:
        single_input['title_length_binned'] = 'biggest'

########
    if word_count < 10:
        single_input.length_binned =  'small'
    elif word_count < 20:
        single_input.length_binned =  'med'
    elif word_count < 40:
        single_input.length_binned =  'big'
    else:
        single_input.length_binned =  'biggest'
########
    # Change Input Value Here!
    shortest = single_input.copy()
    medium = single_input.copy()
    bigger = single_input.copy()
    biggest = single_input.copy()

    shortest.length_binned = 'small'
    medium.length_binned = 'med'
    bigger.length_binned = 'big'
    biggest.length_binned = 'biggest'

    # single_input.video_length = 'small' # etc
    single_input = pd.get_dummies(single_input, columns=['length_binned', 'title_length_binned'])
    shortest = pd.get_dummies(shortest, columns=['length_binned', 'title_length_binned'])
    medium = pd.get_dummies(medium, columns=['length_binned', 'title_length_binned'])
    bigger = pd.get_dummies(bigger, columns=['length_binned', 'title_length_binned'])
    biggest = pd.get_dummies(biggest, columns=['length_binned', 'title_length_binned'])

    single_input = single_input.reindex(columns = X.columns, fill_value=0)
    shortest = shortest.reindex(columns = X.columns, fill_value=0)
    medium = medium.reindex(columns = X.columns, fill_value=0)
    bigger = bigger.reindex(columns = X.columns, fill_value=0)
    biggest = biggest.reindex(columns = X.columns, fill_value=0)

    proba_dict = {}
    proba_dict['not changing'] = rf.predict_proba(single_input)[0][1]
    proba_dict['the shortest'] = rf.predict_proba(shortest)[0][1]
    proba_dict['a medium'] = rf.predict_proba(medium)[0][1]
    proba_dict['a long'] = rf.predict_proba(bigger)[0][1]
    proba_dict['the longest'] = rf.predict_proba(biggest)[0][1]


    length_bin_dict = {}
    length_bin_dict['not changing'] = ""
    length_bin_dict['the shortest'] = "(0-10 words)"
    length_bin_dict['a medium'] = "(10-20 words)"
    length_bin_dict['a long'] = "(20-40 words)"
    length_bin_dict['the longest'] = "(40-200 words)"


    best_proba_key = max(proba_dict, key=proba_dict.get)
    ##### model outputs
    print_1 = '' 

    print_2 =  ((f"{selected_dependent_target_variable} probability with current length: {proba_dict['not changing']}"))

    print_3,print_4,print_5,print_6 ='','','',''

    if not single_input.equals(shortest):
        print_3 = ((f"{selected_dependent_target_variable} probability with the shortest length: {proba_dict['the shortest']}"))
    if not single_input.equals(medium):
        print_4 =  ((f"{selected_dependent_target_variable} probability with a medium length: {proba_dict['a medium']}"))
    if not single_input.equals(bigger):
        print_5 = ((f"{selected_dependent_target_variable} probability with a long length: {proba_dict['a long']}"))
    if not single_input.equals(biggest):
        print_6 =  ((f"{selected_dependent_target_variable} probability with the longest length: {proba_dict['the longest']}"))

    print_7 =  ((f'\nThis model recommmends using {best_proba_key} text length. {length_bin_dict[best_proba_key]}'))
    print_log = ''

    return print_1,print_2,print_3,print_4,print_5,print_6,print_7,print_log
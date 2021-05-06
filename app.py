
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
#####
from custom_functions import parse_features_from_html, make_predictions




app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/submit')
def submit():

    data = request.args

    html = data['html']
    TargetVariable = data['TargetVariable']
    CampaignType = data['CampaignType']
    IndustryType = data['IndustryType']

    soup = BeautifulSoup(html, 'html.parser')



    visible_text,links,img_tags,title_count,word_count,link_count,img_count,main_image = parse_features_from_html(soup)
    




        
    print_1,print_2,print_3,print_4,print_5,print_6,print_7,print_log = make_predictions(target_variable=TargetVariable, 
                                                                                            campaign_type=CampaignType, 
                                                                                            industry_type=IndustryType,
                                                                                            word_count=word_count,
                                                                                            link_count=link_count,
                                                                                            img_count=img_count,
                                                                                            title_count=title_count)  

##########


    return render_template('predictions.html',  html=html,
                                                TargetVariable=TargetVariable,
                                                CampaignType=CampaignType,
                                                IndustryType=IndustryType,
                                                title=visible_text[0],
                                                visible_text=visible_text[1:],
                                                links=[i['href'] for i in links],
                                                img_tags=[i['src'] for i in img_tags],
                                                main_image = main_image['src'],
                                                word_count=word_count,
                                                title_count=title_count,
                                                link_count=link_count,
                                                img_count=img_count,
                                                print_1=print_1,
                                                print_2=print_2,
                                                print_3=print_3,
                                                print_4=print_4,
                                                print_5=print_5,
                                                print_6=print_6,
                                                print_7=print_7,
                                                print_log=print_log
                                                )

if __name__ =='__main__': 
    app.run(debug = True)    
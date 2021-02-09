
from flask import Flask, Response, request, render_template, jsonify
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




app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/submit')
def submit():

    # loading TF-IDF Vectorizer weights:
    model_dir = 'pkl_files/'
    filename = 'tfidf_vectorizer.sav'
    tfidf_vectorizer = joblib.load(model_dir + filename)

    # data here is the response from the get request input from the form template
    # double check what this is pulling from the website
    # print it out as part of the analysis on the app
    # check if this includes the title or not

##### New Stuff Inserted Here!
    data = request.args
    news_article_URL = data['NewsInput']
    user_input_prediction = data['UserPrediction']
    # pulling html from webite with
    html = requests.get(news_article_URL).text
    text = fulltext(html)

    # transforming the news article into TF-IDF Matrix:
    X_test = tfidf_vectorizer.transform([text])
    

    # loading all five models
    
    # loading Naive Bayes Model:
    filename = 'tfidf_nb_classifier.sav'
    tfidf_nb_classifier = joblib.load(model_dir + filename)
    nb_pred = tfidf_nb_classifier.predict(X_test)[0]
    nb_proba = tfidf_nb_classifier.predict_proba(X_test)[0][1]

    # loading the TF-IDF Logistic Regression Model:
    filename = 'tfidf_lr_classifier.sav'
    tfidf_lr_classifier = joblib.load(model_dir + filename)
    lr_pred = tfidf_lr_classifier.predict(X_test)[0]
    lr_proba = tfidf_lr_classifier.predict_proba(X_test)[0][1]

    # loading the TF-IDF Random Forest Model:
    filename = 'tfidf_rf_classifier.sav'
    tfidf_rf_classifier = joblib.load(model_dir + filename)
    rf_pred = tfidf_rf_classifier.predict(X_test)[0]
    rf_proba = tfidf_rf_classifier.predict_proba(X_test)[0][1]

    # loading the TF-IDF Passive Aggressive Classification Model:
    filename = 'tfidf_pac_classifier.sav'
    tfidf_pac_classifier = joblib.load(model_dir + filename)
    pac_pred = tfidf_pac_classifier.predict(X_test)[0]
    pac_pdf = tfidf_pac_classifier.decision_function(X_test)[0]
    pac_proba = stats.norm.cdf(pac_pdf)

    # loading the TF-IDF Support Vector Machine Model:
    filename = 'tfidf_svm_classifier.sav'
    tfidf_svm_classifier = joblib.load(model_dir + filename)
    svm_pred = tfidf_svm_classifier.predict(X_test)[0]
    svm_pdf = tfidf_svm_classifier.decision_function(X_test)[0]
    svm_proba = stats.norm.cdf(svm_pdf)
    
    # this counter with track the overall truthiness of the models:
    fake_news_counter = 0
    
    if nb_pred == "fake":
        fake_news_counter += 1
    if lr_pred == "fake":
        fake_news_counter += 1    
    if rf_pred == "fake":
        fake_news_counter += 1    
    if pac_pred == "fake":
        fake_news_counter += 1    
    if svm_pred == "fake":
        fake_news_counter += 1

    # this variable with be used for logging predictions for the SQL database
    algorithm_pred = ""

    if fake_news_counter <= 3:
          conclusion = "Based on these results, the majority of the models determined this to be REAL news!"
          algorithm_pred = "real"
    else:
          conclusion = "Based on these results, the majority of the models determined this to be FAKE news!" 
          algorithm_pred = "fake"   
              

#####

    # saving data to SQL database

    #db_file = "database/fake_news_database.db"
    DATABASE_URL = os.environ['DATABASE_URL']
    # setting connection variable to None for later try/except statement
    conn = None     
    # the data to be appended to database
    new_data = (news_article_URL, text, user_input_prediction,algorithm_pred)

    query_1 ='''CREATE TABLE IF NOT EXISTS fake_news_table(
                url TEXT PRIMARY KEY,
                article_body TEXT,
                user_pred TEXT,
                algorithm_pred FLOAT);'''

    query_2 = ''' INSERT INTO fake_news_table(url,article_body,user_pred,algorithm_pred)
                    VALUES(%s, %s, %s, %s);'''

    # create a database connection
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        # create a new cursor
        cur = conn.cursor()
        # create table if not exists
        cur.execute(query_1)
        # append data to table
        cur.execute(query_2, new_data)
        # commit the changes to the database
        conn.commit()
        # # close communication with the database
        # cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    # finally:
    #     if conn is not None:
    #         conn.close()

    return render_template('predictions.html', url_submitted = news_article_URL, article_body = text, 
                            user_input_prediction=user_input_prediction, 
                            nb_pred=nb_pred, nb_proba=round(nb_proba,2), 
                            lr_pred=lr_pred, lr_proba=round(lr_proba,2), 
                            rf_pred=rf_pred, rf_proba=round(rf_proba,2), 
                            pac_pred=pac_pred, pac_pdf=round(pac_proba,2), 
                            svm_pred=svm_pred, svm_pdf=round(svm_proba,2), 
                            fake_news_counter=fake_news_counter, 
                            conclusion = conclusion)


if __name__ =='__main__': 
    app.run(debug = True)    
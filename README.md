# Fake_News_Sentiment_Analysis

This project is a work in progress. This project entails training a classification model that uses sentiment anlaysis to determine if a piece of journalism is real or fake news.<br/>

You can check out the deployed Flask app here: [Fake News Detection App](https://fakenewssentimentanalysis.herokuapp.com/)<br/>

I have also created an API endpoint for the Flask app.<br/>
Here is the url to access the API (no access key):'https://fakenewssentimentanalysis.herokuapp.com/todo/api/v1.0/tasks'<br/>

The API accepts a POST argument of a JSON object consisting of two "key:value" pairs, a URL and a string stating if the user believes the article to be real of fake: {"URL":"some_website", "UserPrediction":"real_or_fake"}<br/>
<br/>
<br/>


Below is an example of how to access the API using the Python requests library:<br/>
<br/>
url = 'https://fakenewssentimentanalysis.herokuapp.com/todo/api/v1.0/tasks'<br/>
myobj = {"URL":"https://zapatopi.net/treeoctopus/", "UserPrediction":"fake"}<br/>
r = requests.post(url, json = myobj)<br/>
print(r.text)<br/>

>> {"algorithm_pred":"fake",<br/>
>> "article_body":"\n\nRare photo of the elusive tree octopus\n\n(Enhanced from cropped telephoto) Rare photo of the elusive tree octopus...before eventually moving out of the water and beginning their adult lives.",<br/>
>> "conclusion":"Based on these results, the majority of the models determined this to be FAKE news!",<br/>
>>"fake_news_counter":5,<br/>
>>"lr_pred":"fake","lr_proba":0.28,"nb_pred":"fake","nb_proba":0.41,"pac_pdf":0.41,"pac_pred":"fake",<br/>
>>"rf_pred":"fake","rf_proba":0.38,"svm_pdf":0.38,"svm_pred":"fake",<br/>
>>"url_submitted":`"https://zapatopi.net/treeoctopus/"`, "user_input_prediction":"fake"}<br/>


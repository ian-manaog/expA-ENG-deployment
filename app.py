from flask import Flask, request, render_template
from helpers.utils import *
app = Flask(__name__)

vectorizer,classifier = load_objects() #load the model and vectorizer


@app.route("/")
def index():
    return render_template('form.html')

@app.route('/',methods=['POST']) #prediction on data
def predict():
    t_sentiment = ''
    text = request.form['text'] #input is from forms
    preptext = preprocess(text) #cleaning and preprocessing of the texts

    if preptext: #preptext is true when not empty
        preptext = np.array([preptext])
        vector_text = vectorizer.transform(preptext) #vectorize text
        predictions = classifier.predict(vector_text) #making predictions
        sentiment = int(np.argmax(predictions)) #index of maximum prediction
        probability = max(predictions.tolist()[0]) #probability of maximum prediction
        if sentiment==0: #assigning appropriate name to prediction
            t_sentiment = 'negative'
        elif sentiment==1:
            t_sentiment = 'postive'
        #no neutral class was found in the dataset used for training
        
        return render_template('form.html', sentiment=t_sentiment, text=text)
    #cases where after text preprocessing, output is empty string
    else:
        return render_template('form.html', sentiment='none', text=text)
    
import nltk
from flask import Flask, request, render_template
from helpers.utils import *
app = Flask(__name__)

nltk.download('english')
vectorizer,classifier = load_objects() #load the model and vectorizer


@app.route("/")
def index():
    return render_template('form.html')

@app.route('/',methods=['POST']) #prediction on text
def predict():
    t_sentiment='' #initialize t_sentiment
    text = request.form['text'] #input is from forms
    preptext = preprocess(text) #cleaning and preprocessing of the texts

    #preptext is true when it is not empty
    if preptext:
        preptext = np.array([preptext]) #change to list
        vector_text = vectorizer.transform(preptext) #vectorize text
        predictions = classifier.predict_proba(vector_text) #extracting probability of predicitions
        sentiment = int(np.argmax(predictions)) #index of maximum prediction
        if sentiment==0: #assigning appropriate name to prediction
            t_sentiment = 'negative'
        elif sentiment==1:
            t_sentiment = 'postive'
        #In training the model, there was no neutral class in the dataset so only two classes are available
        
        return render_template('form.html', sentiment=t_sentiment, text=text)
    #cases where after text preprocessing, output is empty string
    else:
        return render_template('form.html', sentiment='none', text=text)
    
if __name__ == "__main__":
    app.run(debug=True)
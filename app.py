# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
lemmatizer=WordNetLemmatizer()
app1=Flask(__name__,template_folder='template')
app=app1

mnb_model=pickle.load(open('models/model_mnb_spam_ham.pkl','rb'))
bow_model=pickle.load(open('models/bag of words .pkl','rb'))

@app.route('/')
def homepage(): 
     return render_template('index.html')
 
@app.route('/find_data',methods=['POST'])
def find_data():
    emoji1 = '😊'
    angry_emoji = '😠'
    if request.method=='POST':
        data=request.form.get('content')
        text=re.sub('[^a-zA-Z]',' ',data)
        text=text.lower()
        text=' '.join(word for word in nltk.word_tokenize(text) if word not in stopwords.words('english'))
        lemm_text=[' '.join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text))]
        bow_form=bow_model.transform(lemm_text).toarray()
        prediction=mnb_model.predict(bow_form)
        if prediction==1:
            return render_template('index.html', results='This content does not seems to be a spam',emoji=emoji1)
        else:
            return render_template('index.html',results='This content seems to be a Spam!!!!',emoji=angry_emoji)
    else:
        return render_template('index.html')
   

if __name__=="__main__":
    app.run(debug=True)

    

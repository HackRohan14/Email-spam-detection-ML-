from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = MultinomialNB()
vectorizer = CountVectorizer()


dataset = pd.read_csv('emails.csv')

X = vectorizer.fit_transform(dataset['text'])
y = dataset['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = [request.form['email']]
        email_features = vectorizer.transform(email)
        prediction = model.predict(email_features)
        result = 'It\'s not Spam!' if prediction[0] == 0 else 'It\'s Spam Mail!'
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

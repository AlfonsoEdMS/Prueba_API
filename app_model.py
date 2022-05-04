from flask import Flask, jsonify, request
import os
import pickle

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for Running a ML Algorithm to Predict Sales Revenue.</p>"
model = pickle.load(open('ad_model.pkl','rb'))

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    
    tv = request.get_json()['tv']
    radio = request.get_json()['radio']
    newspaper = request.get_json()['newspaper']

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
    
    return jsonify({'predictions': prediction[0]})

app.run()
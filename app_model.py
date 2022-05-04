from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for Running a ML Algorithm to Predict Sales Revenue.</p>"

model = pickle.load(open('/home/AlfonsoEdlMS/Prueba_API/ad_model.pkl','rb'))

@app.route('/api/v0/predict', methods=['GET'])
def predict():

    tv = request.get_json()['tv']
    radio = request.get_json()['radio']
    newspaper = request.get_json()['newspaper']

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])

    return jsonify({'predictions': prediction[0]})

@app.route('/api/v0/retrain/', methods=['PUT'])
def retrain():
    archivo = request.get_json()['name']
    data = pd.read_csv('/home/AlfonsoEdlMS/Prueba_API/' + archivo, index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                    data['sales'],
                                                    test_size = 0.20,
                                                    random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('/home/AlfonsoEdlMS/Prueba_API/prueba.pkl', 'wb'))
    # mean = mean_squared_error(y_test, model.predict(X_test))
    # rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return 'Sucess!'

app.run()

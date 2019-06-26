import pickle
from flask import Flask, request, json, jsonify
import numpy as np

app = Flask(__name__)

#filename of saved model
filename = 'diabetes.sav'

#load saved model
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
    #get features to predict
    features = request.json
    
    #create features list for prediction
    features_list = [features["Glucose"], features["BMI"], features["Age"]]
    
    #get prediction class
    prediction = loaded_model.predict([features_list])
    
    #get prediction probabilities
    #confidence = loaded_model.predict_proba([features_list])
    
    #formulate the response to return to client
    response = {}
    response['prediction'] = int(prediction[0])
    #response['confidence'] = str(round(np.amax(confidence[0]) * 100 , 2))
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
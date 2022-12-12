from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# App Initialization
app = Flask(__name__)

# Load the Model
cleaner = pickle.load(open('cleaner.pkl','rb'))
model = tf.keras.models.load_model('model_suicide.h5')

# Endpoint for Homepage
@app.route("/")
def home():
    return "<h1>It Works!</h1>"

# Endpoint for Prediction
@app.route("/predict", methods=['POST'])
def model_predict():
    args = request.json
    new_data = {
    'text': args.get('text')}


    new_data = pd.DataFrame([new_data])
    print('New Data : ', new_data)

    #pipeline
    X = cleaner(new_data)

    # Predict
    y_label = ['No','Yes']
    y_pred = int(np.round(model.predict(X(new_data['preprocessed']))[0][0]))

    # Return the Response
    response = jsonify(
      result = str(y_pred), 
      label_names = y_label[y_pred])

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
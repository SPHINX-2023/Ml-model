from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
import random

app = Flask(__name__)

def lo_model(model_add):
    try:
        model = tf.keras.models.load_model(model_add)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

pre_model = lo_model('exp_03.keras')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400
        input_data = np.array(data['input_data'][0])
        prediction = pre_model.predict(input_data.reshape(1,14,1)) 
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
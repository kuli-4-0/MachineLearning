import json
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from flask import Flask, request

#Load model
model = tf.keras.models.load_model('./content_model')

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, This is ellise!"

@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.json
    test_1 = pd.DataFrame.from_dict(json_data)
    # return json.dumps(json_data)
    ratings = tf.data.Dataset.from_tensor_slices(
        dict(test_1[['id_user', 'id_event', 'nama_event', 'harga_tiket', 'durasi', 'genre', 'id_event_organizer']]))

    ratings = ratings.map(lambda x: {
        'durasi': float(x['durasi']),
        'genre': int(x['genre']),
        'harga_tiket': float(x['harga_tiket']),
        'id_event': x['id_event'],
        'id_event_organizer': x['id_event_organizer'],
        'id_user': x['id_user'],
        'nama_event': x['nama_event'],
        #     'user_rating':float(x['user_rating'])
    })

    cached_test = ratings.batch(4096).cache()

    prediction = model.predict(cached_test)
    prediction_string = [str(d) for d in prediction]

    id_user = test_1['id_user'].values
    id_event = test_1['id_event'].values

    response_json = {
        "id_user": list(id_user),
        "id_event": list(id_event),
        "prediction": list(prediction_string)
    }

    return json.dumps(response_json)

# def api_response():
#     from flask import jsonify
#     if request.method == 'POST':
#         return jsonify(**request.json)

if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000)

import json
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS


#Load model
model = tf.keras.models.load_model('./content_model')

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "Hello, This is ellise!"

@app.route("/predict", methods=["POST"])
def predict():
    # Get Data from user
    data = request.json
    id_user = data['id_user']
    genre = data['genre']

    # for Encode genre manually
    music_genres = {
        'pop': 4,
        'jazz': 3,
        'dangdut': 0,
        'electronic': 1,
        'r&b': 5,
        'rock': 6,
        'indie': 2
    }

    # Get data from database and do encoding
    url = f'https://elise-project.mdwisu.shop/events/getEventForMl?genre={genre}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['data'])
    df['id_user'] = id_user
    df['genre'] = music_genres[genre.lower()]


    # Convert dataframe to tensorflow dataset
    ratings = tf.data.Dataset.from_tensor_slices(
        dict(df[['id_user', 'id_event', 'nama_event', 'harga_tiket', 'durasi', 'genre', 'id_event_organizer']]))

    # Mapping
    ratings = ratings.map(lambda x: {
        'durasi': float(x['durasi']),
        'genre': int(x['genre']),
        'harga_tiket': float(x['harga_tiket']),
        'id_event': str(x['id_event']),
        'id_event_organizer': str(x['id_event_organizer']),
        'id_user': str(x['id_user']),
        'nama_event': x['nama_event'],
    })
    cached_test = ratings.batch(4096).cache()

    # Predicting rating values
    prediction = model.predict(cached_test)

    # Add prediction to df
    df['rating_prediction'] = pd.Series(prediction.flatten())

    # Sort Descending by rating _prediction
    df.sort_values(by='rating_prediction', ascending=False, inplace=True)

    # Df to Json
    # json_data = df.to_json(orient='records')

    return jsonify(df.to_dict(orient='records'))



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

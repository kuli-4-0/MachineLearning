from flask import Flask, request
import pandas as pd
import requests

app = Flask(__name__)

@app.route('/events/<genre>')
def get_events(genre):
    url = f'https://elise-project.mdwisu.shop/events/getEventForMl?genre={genre}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['data'])
    return df.to_string()


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000)

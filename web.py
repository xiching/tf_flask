import tensorflow as tf
import numpy as np

from flask import Flask, jsonify, request,make_response
from flask_cors import CORS

from model_wrappers import ModelA, ModelB
a = ModelA()
b = ModelB()

app = Flask(__name__)
CORS(app)

def featurize1(*args):
    """
    Dummy featurizer for model_a
    """
    model_size = (10, 10)
    return np.arange(100, dtype=np.float32).reshape(model_size)

def featurize2(*args):
    """
    Dummy featurizer for model_a
    """
    model_size = (4, 4)
    return np.random.random(model_size)


@app.route('/get_my_predictions', methods=['POST'])
def get_predictions():
    print(request.json)
    text = request.json.get('text')
    vec1 = featurize1(text)
    vec2 = featurize2(text)
    resp_a = a.predict(vec1).tolist()
    resp_b = b.predict(vec2).tolist()

    return jsonify({"Response":[{"model_a": resp_a}, {"model_b":resp_b}]})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',threaded=True)

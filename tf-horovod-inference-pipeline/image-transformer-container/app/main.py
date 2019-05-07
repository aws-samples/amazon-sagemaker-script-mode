import io
import json
import numpy as np
import struct

from flask import Flask, Response, request
from PIL import Image 


app = Flask(__name__)


def read_image(image_from_s3):
    
    image_as_bytes = io.BytesIO(image_from_s3)
    image = Image.open(image_as_bytes)
    instance = np.expand_dims(image, axis=0)
    
    return instance.tolist()


@app.route("/invocations", methods=['POST'])
def invocations():
    
    try:
      image_for_JSON = read_image(request.data)
      # TensorFlow Serving's REST API requires a JSON-formatted request
      response = Response(json.dumps({"instances": image_for_JSON}))
      response.headers['Content-Type'] = "application/json"
      return response
    except ValueError as err:
      return str(err), 400


@app.route("/ping", methods=['GET'])
def ping():
    
    return "", 200

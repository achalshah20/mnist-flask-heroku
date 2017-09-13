import numpy as np
import sys
from flask import request,jsonify,render_template,Flask
import keras
from keras.models import model_from_json
import logging
# webapp
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
model = keras.models.load_model('final_model.h5')

@app.route('/mnist', methods=['POST'])
def mnist():
    input = (np.array(request.json, dtype=np.uint8) / 255.0).reshape(1,28, 28,1)
    output1 = model.predict(input)
    return jsonify(results=output1.tolist())


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

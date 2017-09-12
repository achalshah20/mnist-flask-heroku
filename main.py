import numpy as np
from flask import request,jsonify,render_template,Flask
import keras
# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    model = keras.models.load_model('model.json')
    output1 = model.predict(input)
    return jsonify(results=[output1])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

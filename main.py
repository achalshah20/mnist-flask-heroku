import numpy as np
import sys
from flask import request,jsonify,render_template,Flask
import keras
from keras.models import model_from_json

# webapp
app = Flask(__name__)


@app.route('/mnist', methods=['POST'])
def mnist():
    print("received mnist requirest")
    sys.stdout.flush()
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1,28, 28,1)
    model = keras.models.load_model('final_model.h5')
    output1 = model.predict(input)
    print(output1)
    print("predicted model")
    sys.stdout.flush()
    return jsonify(results=output1.tolist())


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

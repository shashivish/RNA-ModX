import pickle
import json
from model_files.ml_model import predict_rna_modification_status

from flask import Flask, jsonify, request

##creating a flask app and naming it "app"
app = Flask('app')


@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'


@app.route('/predict', methods=['POST'])
def predict():
    rna_sequence = request.get_json()
    print("---------------------------------------------")
    print(rna_sequence)
    print("---------------------------------------------")

    # data = json.loads(rna_sequence)
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_rna_modification_status(rna_sequence['input'], model)

    for p in predictions:
        r = int(p)
    result = {
        'model_prediction': r
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

import pickle
import json
from model_files.ml_model import ModelPredictionHelperClass

from flask import Flask, jsonify, request, render_template, flash

##creating a flask app and naming it "app"
app = Flask('app')
app.config['SECRET_KEY'] = 'mysecretkey'
app.jinja_env.globals.update(len=len)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET', 'POST'])
def test():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        MODEL_FILE_NAME = 'xgboost_model.bin'
        rna_sequence = request.form.get('RNA Sequence')
        window_size = request.form.get('Window Size')

        if not rna_sequence:
            flash('RNA Sequence is required!')
        else:
            # data = json.loads(rna_sequence)
            with open('./model_files/' + MODEL_FILE_NAME, 'rb') as f_in:
                model = pickle.load(f_in)
                f_in.close()

            print("Model Loaded Successfully.")
            cls = ModelPredictionHelperClass()
            predictions = cls.predict_rna_modification_status(rna_sequence, model, window_size)


            # return jsonify(result)
            return render_template('prediction.html', rna_sequence=rna_sequence , position=predictions)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

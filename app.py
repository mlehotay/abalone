from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    new = pd.DataFrame({
        'Sex': [args.get('sex')],
        'Length': [args.get('length')],
        'Diameter': [args.get('diameter')],
        'Height': [args.get('height')],
        'Whole weight': [args.get('whole')],
        'Shucked weight': [args.get('shucked')],
        'Viscera weight': [args.get('viscera')],
        'Shell weight': [args.get('shell')]
    })
    prediction = int(round(pipe.predict(new)[0]))
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()

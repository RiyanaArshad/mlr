import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])

        # Make prediction
        features = np.array([[size, bedrooms, age]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f'Estimated Price: ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', error_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)

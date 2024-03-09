from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pixel_array = data.get('pixelArray')  # Correct key name here

    payload = {'imagedata': pixel_array}  # Correct key name here
    response = requests.post('http://127.0.0.1:5001/predict', json=payload)
    
    # Check response
    if response.status_code == 200:
        prediction = response.json()['prediction']  # Get the prediction
        output = response.json()['output']  # Get the prediction
        
        return jsonify({'prediction': prediction , 'output':output})
    else:
        return jsonify({'error': 'Prediction request failed'})

if __name__ == '__main__':
    app.run(debug=True)

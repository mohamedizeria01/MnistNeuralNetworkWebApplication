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


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    pixel_array = data.get('pixelArray')  # Retrieve pixel array
    correct_label = data.get('correctLabel')  # Retrieve the correct label

    if pixel_array is None or correct_label is None:
        return jsonify({'error': 'Invalid data provided'}), 400

    payload = {
        'imagedata2': pixel_array,
        'label': correct_label  # Send corrected label
    }

    # Send feedback data to the model training service
    response = requests.post('http://127.0.0.1:5001/feedback', json=payload)

    # Check response status
    if response.status_code == 200:
        return jsonify({'message': 'Feedback received and weights updated'})
    elif response.status_code == 400:
        return jsonify({'message': 'Weights remain unchanged'})
    else:
        return jsonify({'error': 'Feedback request failed'}), 500


if __name__ == '__main__':
    app.run(debug=True)

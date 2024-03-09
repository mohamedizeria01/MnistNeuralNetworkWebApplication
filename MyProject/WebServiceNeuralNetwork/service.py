import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Function to load weights from pkl file
def load_weights(file_path):
    with open(file_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

# Load the weights back
loaded_weights = load_weights('model_weights.pkl')

def Relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = Relu(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, axis=0)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions , A2

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def test_prediction():
    outp=None
    if request.method == 'POST':
        data = request.get_json()
        current_image = np.array(data['imagedata'])  # Assuming 'image' is a list or array
        prediction,outp = make_predictions(current_image, loaded_weights['W1'], loaded_weights['b1'], loaded_weights['W2'], loaded_weights['b2'])
        return jsonify({'prediction': prediction.tolist(), 'output': outp.tolist()})
        
        # print(current_image)
        


if __name__ == '__main__':
    app.run(debug=True, port=5001)

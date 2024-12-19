import numpy as np
from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
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
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X): 
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

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


import numpy as np
import pickle
import os

def update_weights_with_feedback(pixel_array, correct_label, alpha=2.0, beta=0.0):
    """
    Updates the model weights using Stochastic Gradient Descent (SGD) with a large learning rate.

    Args:
        pixel_array (numpy.ndarray): The input image as a 1D array (28x28 flattened to 784).
        correct_label (int): The correct label for the image.
        alpha (float): Learning rate (set to a larger value).
        beta (float): Momentum coefficient (not used in SGD).

    Returns:
        str: Message indicating if the weights were updated or not.
    """
    # Load existing weights
    if os.path.exists('model_weights.pkl'):
        with open('model_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    else:
        raise FileNotFoundError("Model weights file not found.")

    # Prepare input and label
    X = pixel_array.reshape(-1, 1)  # Reshape to (784, 1)
    one_hot_Y = np.zeros((10, 1))
    one_hot_Y[correct_label] = 1  # One-hot encode the correct label

    # Forward propagation
    Z1 = W1.dot(X) + b1
    A1 = np.maximum(Z1, 0)  # ReLU activation
    Z2 = W2.dot(A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2))  # Softmax activation

    # Backward propagation (gradients)
    dZ2 = A2 - one_hot_Y
    dW2 = dZ2.dot(A1.T)
    db2 = dZ2
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)  # ReLU derivative
    dW1 = dZ1.dot(X.T)
    db1 = dZ1

    # Update weights using SGD (without momentum)
    new_W1 = W1 - alpha * dW1
    new_b1 = b1 - alpha * db1
    new_W2 = W2 - alpha * dW2
    new_b2 = b2 - alpha * db2

    # Check if the weights have changed
    if np.any(W1 != new_W1) or np.any(b1 != new_b1) or np.any(W2 != new_W2) or np.any(b2 != new_b2):
        weights_updated = True
        message = 1
    else:
        weights_updated = False
        message = 0

    # Save updated weights if changed
    if weights_updated:
        updated_weights = {'W1': new_W1, 'b1': new_b1, 'W2': new_W2, 'b2': new_b2}
        with open('model_weights.pkl', 'wb') as f:
            pickle.dump(updated_weights, f)

    return message  # Indicating if the weights were updated


import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    try:
        data = request.json
        logging.info("Received data: %s", data)

        # Validate pixel array and label
        pixel_array = np.array(data['imagedata2'])
        if pixel_array.size != 784:
            raise ValueError("Pixel array must have exactly 784 elements (28x28 flattened image).")
        correct_label = int(data['label'])
        if not (0 <= correct_label <= 9):
            raise ValueError("Label must be an integer between 0 and 9.")

        logging.info("Pixel Array Shape: %s", pixel_array.shape)
        logging.info("Correct Label: %d", correct_label)

        # Update weights
        msg=update_weights_with_feedback(pixel_array, correct_label, alpha=0.5, beta=0.1)

        if msg==1:
            return jsonify({'message': 'Weights updated successfully'}), 200
    
        return jsonify({'message':  "Weights remain unchanged"}), 400
    


    except ValueError as ve:
        logging.error("Validation Error: %s", str(ve))
        return jsonify({'error': str(ve)}), 400  # Bad Request

    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return jsonify({'error': str(e)}), 500  # Internal Server Error



if __name__ == '__main__':
    app.run(debug=True, port=5001)

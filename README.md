# Letter Picture Prediction

This project is designed for letter classification using 8x8 grayscale images. A simple neural network is used to predict which letter is represented in the image. The project includes a graphical user interface (GUI) built using Tkinter, allowing users to select images and display prediction results along with the prediction accuracy.

## Features
- Converts RGB images to grayscale.
- Binary conversion of grayscale images.
- Neural network-based letter classification.
- Graphical User Interface (GUI) for image selection and displaying prediction results.
- Displays both the predicted letter and confidence score.

## How it Works
1. **Preprocessing:**
    - Input images are converted to grayscale and then binarized.
    - Images are flattened to 64 features (8x8 pixels) to be used as input to the neural network.
    
2. **Neural Network:**
    - The network consists of:
      - Input layer (64 nodes)
      - Hidden layer (100 nodes, adjustable)
      - Output layer (number of classes/letters)
    - Sigmoid activation function is used.
    - Weights are updated using backpropagation and the learning rate is set to 0.5.
    
3. **Training:**
    - The model is trained on labeled data with one-hot encoding for target labels.
    
4. **Prediction:**
    - The trained model is used to predict the letter and its confidence when a new image is selected from the GUI.

## Installation
### Requirements
To run this project, you need to install the following libraries:

```bash
pip install numpy matplotlib pillow

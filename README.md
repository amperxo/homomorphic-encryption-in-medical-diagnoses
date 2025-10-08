1. Libraries

torch, torchvision: for neural network and image preprocessing.

tenseal: for homomorphic encryption.

gradio: for creating a simple web interface.

PIL & numpy: for image handling and numeric operations.

2. Model Definition — HECNN

A simple CNN model is defined with:

Two convolution layers (conv1, conv2) + average pooling.

One hidden fully connected layer (fc1) and an output layer (fc2).

The forward pass stops at fc1 for encryption.

The model predicts four classes:
["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"].

3. Model Loading

Loads pre-trained model weights from covid_radiograph_model_final.pth.

Sets the model to evaluation mode (model.eval()).

4. Image Preprocessing

Each uploaded X-ray image is:

Resized to 64×64 pixels

Converted to grayscale

Normalized and converted into a PyTorch tensor.

5. Encrypted Inference

The image is passed through the model up to the first fully connected layer (fc1).

The numeric output is encrypted using TenSEAL’s CKKS scheme (supports encrypted real-number arithmetic).

The final layer (fc2) computations — matrix multiplication and bias addition — are done on the encrypted data.

No decryption happens during computation — ensuring complete privacy.

6. Decryption & Prediction

The encrypted output is decrypted after inference.

A softmax function converts scores to probabilities.

The highest probability class is chosen as the final prediction.

7. User Interface

A Gradio interface allows users to:

Upload a chest X-ray.

See the predicted disease and class probabilities securely.

The interface title:

“Privacy-Preserving COVID-19 X-ray Classifier — Encrypted inference using TenSEAL + PyTorch.”

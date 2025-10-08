import gradio as gr
import tenseal as ts
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Class names
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Model definition (same as training)
class HECNN(torch.nn.Module):
    def __init__(self):
        super(HECNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 4)

    def forward_up_to_fc1(self, x):
        x = self.conv1(x)
        x = x * x
        x = self.pool(x)
        x = self.conv2(x)
        x = x * x
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = x * x
        return x

# Load model and weights
model = HECNN()
model.load_state_dict(torch.load("covid_radiograph_model_final.pth", map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Gradio prediction function
def predict(image: Image.Image):
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # Get fc1 output
    with torch.no_grad():
        fc1_output = model.forward_up_to_fc1(input_tensor).squeeze().numpy()

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 20, 40]
    )
    context.generate_galois_keys()
    context.global_scale = 2**20

    # Encrypt fc1 output
    enc_input = ts.ckks_vector(context, fc1_output.tolist())

    # Run encrypted fc2 inference
    fc2_weight = model.fc2.weight.detach().numpy()
    fc2_bias = model.fc2.bias.detach().numpy()

    enc_output = []
    for i in range(4):
        weight_row = fc2_weight[i]
        bias_val = fc2_bias[i]
        enc_result = enc_input.dot(weight_row)
        bias_vec = ts.ckks_vector(context, [bias_val])
        enc_result = enc_result + bias_vec
        enc_output.append(enc_result)

    # Decrypt and apply softmax
    scores = [x.decrypt()[0] for x in enc_output]
    probs = torch.softmax(torch.tensor(scores), dim=0)
    predicted_index = torch.argmax(probs).item()

    # Prepare output
    class_probs = {class_names[i]: float(probs[i]) for i in range(4)}
    return class_names[predicted_index], class_probs

# Launch Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.JSON(label="Class Probabilities")
    ],
    title="Privacy-Preserving COVID-19 X-ray Classifier",
    description="Encrypted inference using TenSEAL + PyTorch. Upload an X-ray to classify securely."
).launch()

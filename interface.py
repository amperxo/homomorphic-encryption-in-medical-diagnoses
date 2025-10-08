import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
import cv2
from PIL import Image

# Define model
class HECNN(nn.Module):
    def __init__(self):
        super(HECNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        x = self.pool(x)
        x = self.conv2(x)
        x = x * x
        self.feature_maps = x
        self.feature_maps.retain_grad()  # 🔥 Retain grad for Grad-CAM
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HECNN().to(device)
model.load_state_dict(torch.load("covid_radiograph_model_final.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Grad-CAM function
def generate_heatmap(image_tensor, class_idx):
    model.zero_grad()
    output = model(image_tensor)
    class_score = output[0, class_idx]
    class_score.backward()

    gradients = model.feature_maps.grad[0].cpu().numpy()
    activations = model.feature_maps.detach()[0].cpu().numpy()

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (64, 64))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam
    return cam

# Overlay heatmap
def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, output_size=(256, 256)) -> Image.Image:
    img = img.resize((64, 64)).convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
    overlayed_img = Image.fromarray(overlayed)
    # Resize output image to larger size
    overlayed_img = overlayed_img.resize(output_size, resample=Image.BILINEAR)
    return overlayed_img


# Inference + Grad-CAM
def predict_with_heatmap(image):
    original = image.copy()
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_tensor.requires_grad_()  # 🧠 Needed for backprop

    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    label = class_names[predicted.item()]

    heatmap = generate_heatmap(image_tensor, predicted.item())
    result_image = overlay_heatmap(original, heatmap)

    return label, result_image

# Gradio Interface
interface = gr.Interface(
    fn=predict_with_heatmap,
    inputs=gr.Image(type="pil"),
    outputs=["label", "image"],
    title="COVID-19 X-ray Classifier with Grad-CAM",
    description="Upload a chest X-ray to classify and see where the model focused."
)

if __name__ == "__main__":
    interface.launch()

import streamlit as st
from model import VisionTransformer
import torch
import numpy as np
from PIL import Image

# Dictionary of trained drug classes
drugs = {
    0: 'Bioflu',
    1: 'Biogesic',
    2: 'Bactidol',
    3: 'Alaxan',
    4: 'Medicol',
    5: 'Fish Oil',
    6: 'DayZinc',
    7: 'Decolgen',
    8: 'Kremil S',
    9: 'Neozep'
}

state_dict = torch.load("model_weights joleen.pth", map_location=torch.device('cpu'))

model = VisionTransformer()
model.load_state_dict(state_dict)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((72, 72))
    image = np.array(image)
    image = torch.tensor(image)
    image = image.float()
    image = image.reshape((1, 72, 72, 3))
    return image

# Streamlit app
def main():
    st.title("Automated Drug Authentication System")

    # Check GPU availability
    if torch.cuda.is_available():
        st.write("GPU is available")
        device = torch.device("cuda")
    else:
        st.write("GPU is not available, using CPU")
        device = torch.device("cpu")

    # Move the model to the appropriate device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button('Predict'):
            image_array = preprocess_image(image)
            with torch.no_grad():
                outputs = model(image_array)
                predicted_class = torch.argmax(outputs, dim=1).item()
                
                # Check if the predicted class is within the trained drug dictionary
                if predicted_class not in drugs:
                    raise ValueError("Fake drug")
                
                st.write(f"Predicted Class: {drugs[predicted_class]}")

if __name__ == "__main__":
    main()

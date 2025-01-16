import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
model = load_model("malaria-cnn-v3.keras")

# Function to generate Grad-CAM
def generate_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    gradcam = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)

    # Normalize the Grad-CAM map to range [0, 1]
    gradcam = tf.maximum(gradcam, 0)  # Apply ReLU (max with 0)
    gradcam = gradcam / tf.reduce_max(gradcam)  # Normalize to [0, 1]
    
    # Invert the Grad-CAM map to ensure high-intensity areas are red
    gradcam = 1 - gradcam  # Invert the values

    # Convert the Grad-CAM to a heatmap in the range [0, 255] for visualization
    gradcam = np.uint8(255 * gradcam)  # Convert to [0, 255] range
    gradcam = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)  # Apply color map for better visualization
    
    return gradcam


# Function to overlay Grad-CAM on the original image
def overlay_gradcam(original_img, gradcam, alpha=0.5):
    # Resize the Grad-CAM heatmap to match the original image size
    heatmap = cv2.resize(gradcam, (original_img.shape[1], original_img.shape[0]))
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img


# Streamlit Dashboard
st.title("Malaria Image Classification with Grad-CAM")
st.markdown(
    '<span style="font-size:20px; font-style:italic; color:black;">Created by ModelBliss (2025)</span>',
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image using PIL and resize it
    image = Image.open(uploaded_file)
    original_img = np.array(image.resize((224, 224))) / 255.0  # Resize and normalize to [0, 1]
    img_array = np.expand_dims(original_img, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = 1 if prediction[0][0] >= 0.5 else 0
    class_label = "Uninfected" if predicted_class == 1 else "Parasitized"
    
    # Display prediction
    st.write(f"Prediction: {class_label}")

    # Generate Grad-CAM
    layer_name = "conv2d_8"  # Layer name for Grad-CAM
    gradcam = generate_gradcam(model, img_array, layer_name)

    # Overlay Grad-CAM
    original_img_uint8 = (original_img * 255).astype(np.uint8)  # Convert to uint8 for visualization
    overlayed_img = overlay_gradcam(original_img_uint8, gradcam)

    # Create columns to display images in the same row
    col1, col2, col3 = st.columns(3)

    # Display the images in the same row with equal width and height
    with col1:
        st.image(original_img, caption="Original Image", use_container_width=True)

    with col2:
        st.image(gradcam, caption="Grad-CAM Heatmap", use_container_width=True)

    with col3:
        st.image(overlayed_img, caption="Overlayed Image", use_container_width=True)

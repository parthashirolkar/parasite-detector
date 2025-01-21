import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


@st.cache_resource
def load_assets():
    with open("class_label_mappings.json", "r") as f:
        class_mappings = json.load(f)
    class_mappings = {v: k for k, v in class_mappings.items()}

    tf_model = tf.lite.Interpreter(model_path="quantized_model.tflite")
    tf_model.allocate_tensors()
    tf_full_model = tf.keras.models.load_model("parasite-detector.h5")

    return class_mappings, tf_model, tf_full_model


label_mappings, model, full_model = load_assets()


def preprocess_image(image) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def predict(image: np.ndarray) -> str:
    processed_image = preprocess_image(image)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], processed_image)
    model.invoke()  
    prediction = model.get_tensor(output_details[0]['index'])
    prediction = prediction.argmax(1)
    res = label_mappings[prediction[0]]
    return res


def get_top_10_feature_maps(image: np.ndarray, layer_index: int, full_model):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Create a model to extract the feature maps from the desired layer
    # layer_model = tf.keras.Model(inputs=full_model.layers[0].input, outputs=full_model.layers[layer_index].output)
    layer_model = tf.keras.models.Sequential([layer for layer in full_model.layers[:layer_index + 1]])

    # Predict to get the feature maps
    feature_maps = layer_model.predict(processed_image, verbose=0)[0]

    # Calculate the mean activation for each feature map
    mean_activations = np.mean(feature_maps, axis=(0, 1))

    # Get the indices of the top 10 feature maps
    top_indices = np.argsort(mean_activations)[-10:][::-1]

    # Select the top 10 feature maps
    top_feature_maps = feature_maps[:, :, top_indices]

    return top_feature_maps




st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Feature Map Visualization"])

st.title("Cell Type Classifier")
st.write("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    if page == "Prediction":
        with col1:
            st.header("Uploaded Image")
            st.image(image, channels="RGB", caption='Uploaded Image.', use_container_width=True)

        with col2:
            st.header("Prediction")
            if st.button("Predict"):
                with st.spinner("Classifying..."):
                    prediction = predict(image)
                    st.write("Prediction:", prediction)

    elif page == "Feature Map Visualization":
        with col1:
            st.header("Uploaded Image")
            st.image(image, channels="RGB", caption='Uploaded Image.', use_container_width=True)

        with col2:
            st.header("Feature Map Visualization")

            # Get a list of convolutional layers
            conv_layers = [layer.name for layer in full_model.layers if 'conv' in layer.name]
            layer_index = st.selectbox("Select convolutional layer", range(len(conv_layers)),
                                       format_func=lambda x: conv_layers[x])

            if st.button("Show Feature Maps"):
                with st.spinner("Extracting feature maps..."):
                    top_feature_maps = get_top_10_feature_maps(image, layer_index, full_model)

                    num_feature_maps = top_feature_maps.shape[-1]
                    num_cols = 5  # Number of columns for the feature maps
                    num_rows = (num_feature_maps + num_cols - 1) // num_cols  # Calculate rows needed

                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
                    axes = axes.flatten()

                    for i in range(num_feature_maps):
                        ax = axes[i]
                        ax.imshow(top_feature_maps[:, :, i], cmap='viridis')
                        ax.axis('off')

                    for i in range(num_feature_maps, len(axes)):
                        axes[i].axis('off')

                    with st.expander("Feature Maps"):
                        st.pyplot(fig)

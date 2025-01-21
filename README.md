# Parasite Detection Project

This project provides a deep learning model for detecting and classifying various parasites in microscopic images.

## Dataset Citation

```
@misc{ahmed_alrefaei_2023,
    title={🔬Parasite Dataset: Leishmania, Plasmodium&Babesia},
    url={https://www.kaggle.com/ds/4216937},
    DOI={10.34740/KAGGLE/DS/4216937},
    publisher={Kaggle},
    author={Ahmed Alrefaei},
    year={2023}
}
```

## Project Structure

The project consists of the following files:

- `detect_app.py`: A Streamlit application that allows users to upload an image and classify the cell type using a pre-trained deep learning model. It also provides feature map visualization capabilities.
- `model_quantization.py`: A script responsible for quantizing a Keras Sequential model into a TFLite model, reducing the model size from 148MB to around 25MB while maintaining FP32 accuracy.
- `class_label_mappings.json`: A JSON file containing the mapping between class labels and their corresponding integer values for the classification task.
- `requirements.yaml`: A YAML configuration file specifying the required dependencies for the project.

## Usage

To interact with the model using the Streamlit app, follow these steps:

1. Ensure you have all the required dependencies installed. You can use the `requirements.yaml` file to set up your environment.

2. Launch the Streamlit app by running the following command in your terminal:

   ```
   streamlit run detect_app.py
   ```

3. Once the app is running, you can interact with it through your web browser. The app provides two main functionalities:

   a. Prediction:
   - Upload an image using the file uploader.
   - Click the "Predict" button to classify the cell type in the image.
   - The predicted cell type will be displayed on the screen.

   b. Feature Map Visualization:
   - Upload an image using the file uploader.
   - Select a convolutional layer from the dropdown menu.
   - Click the "Show Feature Maps" button to visualize the top 10 feature maps from the selected layer.

4. You can switch between the "Prediction" and "Feature Map Visualization" pages using the sidebar navigation.

The app provides an intuitive interface for both classifying parasite images and exploring the internal representations learned by the model.
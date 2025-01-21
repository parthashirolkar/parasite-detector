"""This script was made to quantize the keras.Sequential model that can be trained from the notebook
into a TFLite model. This quantiaztion step was taken to reduce the model size from 148MB to around 25MB.
This quantization step also uses FP32 accuracy."""


import tensorflow as tf


def quantize_model():
    # Load the Keras model from the HDF5 file
    model = tf.keras.models.load_model("parasite-detector.keras")

    # Convert to TFLite model with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_quant_model = converter.convert()

    # Save the quantized model
    with open("quantized_model.tflite", "wb") as f:
        f.write(tflite_quant_model)

def main():
    quantize_model()

if __name__ == "__main__":
    main()

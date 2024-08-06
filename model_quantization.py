"""This script was made to quantize the keras.Sequential model that can be trained from the notebook
into a TFLite model. This quantiaztion step was taken to reduce the model size from 148MB to around 25MB.
This quantization step also uses FP32 accuracy."""


import tensorflow as tf


def quantize_model():
    converter = tf.lite.TFLiteConverter.from_saved_model("parasite-detector-model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_quant_model = converter.convert()

    with open("quantized_model.tflite", "wb") as f:
        f.write(tflite_quant_model)

def main():
    quantize_model()

if __name__  == "__main__":
    main()

"""
Export a trained Keras classifier to ONNX format.

Usage:
    python scripts/export_classifier_onnx.py
    python scripts/export_classifier_onnx.py --input models/classifier/best_classifier_model.keras

The matching *_classes.json file is copied next to the output .onnx file
automatically, so inference can find it with the standard path convention.
"""
import argparse
import shutil

import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Export Keras classifier to ONNX")
    p.add_argument("--input", default="models/classifier/classifier_model.keras", help="Path to .keras model")
    p.add_argument("--output", default="models/classifier/classifier_model.onnx", help="Output .onnx path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading Keras model from {args.input} ...")
    model = load_model(args.input)
    print(f"Input shape: {model.input_shape}")

    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
    print(f"Converting to ONNX (opset={args.opset}) ...")
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=args.opset, output_path=args.output)
    print(f"Saved ONNX model → {args.output}")

    classes_src = args.input.replace(".keras", "_classes.json")
    classes_dst = args.output.replace(".onnx", "_classes.json")
    shutil.copy(classes_src, classes_dst)
    print(f"Copied classes JSON → {classes_dst}")


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

import numpy as np
import cv2
import os
import argparse


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path, target_size=(256, 256)):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # normalize instead of preprocess_input (since your model is custom)
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image_array, category_index, layer_name):
    """
    Generate Grad-CAM heatmap - Fixed for TensorFlow 2.x compatibility
    """
    # Ensure image_array is a tensor
    image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)
    
    # Get the target convolutional layer
    conv_layer = input_model.get_layer(layer_name)

    # Create a model that outputs both conv features and predictions
    grad_model = Model(
        inputs=input_model.input,
        outputs=[conv_layer.output, input_model.output]
    )

    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs, predictions = grad_model(image_array, training=False)
        
        # Handle list/tuple outputs
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # Get the score for the target class
        class_channel = predictions[:, category_index]

    # Compute gradients of score w.r.t conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        # Try alternative approach - use the simplified method
        print("⚠️ Gradients are None, using simplified Grad-CAM...")
        return simplified_gradcam(input_model, image_array, category_index, layer_name)

    # Compute pooled gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()

    # Weight conv feature maps by importance
    heatmap = np.dot(conv_outputs, pooled_grads.numpy())

    # Normalize to [0,1]
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Resize heatmap to match input image
    heatmap = cv2.resize(heatmap, (image_array.shape[2], image_array.shape[1]))

    # Create overlay visualization
    img = image_array[0, :].numpy()
    img = (img * 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Blend with original image
    cam = cv2.addWeighted(img, 0.6, cam, 0.4, 0)
    
    return cam, heatmap


def simplified_gradcam(input_model, image_array, category_index, layer_name):
    """
    Simplified Grad-CAM that uses feature map activations directly.
    Fallback when gradient-based method fails.
    """
    # Get the target convolutional layer
    conv_layer = input_model.get_layer(layer_name)

    # Create a model that outputs the conv layer
    conv_model = Model(
        inputs=input_model.input,
        outputs=conv_layer.output
    )

    # Get conv outputs
    conv_outputs = conv_model(image_array, training=False)
    
    # Get predictions to find the target class
    predictions = input_model(image_array, training=False)
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    # Use the feature maps directly (simplified approach)
    # Take the mean across all channels for a simple heatmap
    heatmap = tf.reduce_mean(conv_outputs[0], axis=-1).numpy()

    # Normalize to [0,1]
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Resize heatmap to match input image
    heatmap = cv2.resize(heatmap, (image_array.shape[2], image_array.shape[1]))

    # Create overlay visualization
    img = image_array[0, :].numpy()
    img = (img * 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Blend with original image
    cam = cv2.addWeighted(img, 0.6, cam, 0.4, 0)
    
    return cam, heatmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM of CNN features")
    parser.add_argument("img_dir", type=str, help="Path to target images")
    parser.add_argument("save_dir", type=str, help="Path to save Grad-CAM images")
    parser.add_argument("model_path", type=str, help="Path to .h5 file")
    parser.add_argument("layer_name", type=str, help="Target layer that outputs feature")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, compile=False)

    # ✅ Warm-up call (important for Sequential models)
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    _ = model.predict(dummy_input)

    # ✅ Convert Sequential → Functional if needed
    if isinstance(model, tf.keras.Sequential):
        model = Model(inputs=model.inputs, outputs=model.outputs)

    print(model.summary())

    os.makedirs(args.save_dir, exist_ok=True)

    for fname in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, fname)
        preprocessed_input = load_image(img_path, target_size=(256, 256))

        predictions = model.predict(preprocessed_input)
        predicted_class = np.argmax(predictions)

        cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, args.layer_name)
        out_path = os.path.join(args.save_dir, f"{fname}_{args.layer_name}_gradcam.jpg")
        cv2.imwrite(out_path, cam)
        print(f"✅ Grad-CAM of {fname} saved to {out_path}")

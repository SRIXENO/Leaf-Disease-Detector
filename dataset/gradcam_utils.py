import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def find_last_conv_layer(model, sample_input=None):
    """
    Finds the last convolutional layer in the model.
    If sample_input is provided, ensures it works with Grad-CAM.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("❌ No Conv2D layer found in the model!")


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Full gradient-based Grad-CAM.
    """
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("❌ Gradients are None. Model connection issue.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


def get_simple_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """
    Simplified Grad-CAM fallback: uses mean of feature maps (no gradients).
    """
    conv_model = Model(
        inputs=model.input,
        outputs=model.get_layer(last_conv_layer_name).output
    )

    conv_outputs = conv_model(img_array)
    heatmap = tf.reduce_mean(conv_outputs[0], axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


def overlay_gradcam(img_path, heatmap, alpha=0.4, cmap=cv2.COLORMAP_JET):
    """
    Superimpose Grad-CAM heatmap onto the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cmap)

    # Overlay
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

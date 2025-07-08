import os
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from django.core.files.storage import default_storage

from detector.forms import UploadImageForm

# Load model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'dr_resnet50_model.h5')
model = load_model(model_path)
diagnosis_mapping = {
    0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR"
}
# ------------------ Preprocessing ------------------
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
def circle_crop(img, sigmaX=30):
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    x, y, r = w // 2, h // 2, min(w, h) // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), r, 1, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = crop_image_from_gray(img)
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
# ------------------ Grad-CAM ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, threshold=0.2):
    # Resize the heatmap to match the size of the original image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Normalize heatmap and convert to color map
    heatmap_resized = np.uint8(255 * heatmap_resized)
    color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    # Thresholding to show only strong activations
    heatmap_resized = np.where(heatmap_resized > threshold * 255, heatmap_resized, 0)
    color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    # Overlay the heatmap on the original image
    return cv2.addWeighted(img, 1 - alpha, color_map, alpha, 0)

# ------------------ Prediction Pipeline ------------------
def preprocess_and_predict(image_path, image_name):
    img = cv2.imread(image_path)
    preprocessed = circle_crop(img)
    resized = cv2.resize(preprocessed, (224, 224))
    img_array = np.expand_dims(resized / 255.0, axis=0)
    # Get prediction probabilities
    predictions = model.predict(img_array)[0]
    pred_index = np.argmax(predictions)
    heatmap, _ = make_gradcam_heatmap(img_array, model)
    gradcam = overlay_heatmap(resized.copy(), heatmap)
    # Save images
    pre_path = os.path.join(settings.MEDIA_ROOT, f'preprocessed_{image_name}')
    gradcam_path = os.path.join(settings.MEDIA_ROOT, f'gradcam_{image_name}')
    cv2.imwrite(pre_path, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
    cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR))
    # Format predictions as dictionary with labels and percentages
    probability_dict = {
        diagnosis_mapping[i]: f"{prob*100:.2f}%" for i, prob in enumerate(predictions)
    }
    preds = model.predict(img_array)[0]
    probs = {diagnosis_mapping[i]: f"{p * 100:.2f}%" for i, p in enumerate(preds)}
    return diagnosis_mapping[pred_index], f'preprocessed_{image_name}', f'gradcam_{image_name}', probs
# ------------------ Django View ------------------
def detect(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = request.FILES['image']
            path = default_storage.save('uploaded/' + img_file.name, img_file)
            full_path = os.path.join(settings.MEDIA_ROOT, path)

            prediction, preprocessed_file, gradcam_file, probabilities = preprocess_and_predict(full_path, img_file.name)

            return render(request, 'detector/index.html', {
                'original': path,
                'preprocessed': preprocessed_file,
                'gradcam': gradcam_file,
                'prediction': prediction,
                'probabilities': probabilities,
                'MEDIA_URL': settings.MEDIA_URL
            })
    else:
        form = UploadImageForm()
    return render(request, 'detector/index.html', {'form': form})

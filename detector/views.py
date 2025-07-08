import os
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from django.core.files.storage import default_storage
from django.http import JsonResponse
import logging

from detector.forms import UploadImageForm

# Configure logging
logger = logging.getLogger(__name__)

# Global model variable for lazy loading
model = None

def get_model():
    """Lazy load the model to avoid timeout during startup"""
    global model
    if model is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'model', 'dr_resnet50_model.h5')
            logger.info(f"Loading model from: {model_path}")
            model = load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    return model

# Diagnosis mapping
diagnosis_mapping = {
    0: "No DR", 
    1: "Mild DR", 
    2: "Moderate DR", 
    3: "Severe DR", 
    4: "Proliferative DR"
}

# ------------------ Preprocessing ------------------
def crop_image_from_gray(img, tol=7):
    """Remove black borders from retinal images"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def circle_crop(img, sigmaX=30):
    """Apply circular crop and preprocessing to retinal images"""
    try:
        img = crop_image_from_gray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        x, y, r = w // 2, h // 2, min(w, h) // 2
        
        # Create circular mask
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (x, y), r, 1, -1)
        
        # Apply mask
        img = cv2.bitwise_and(img, img, mask=mask)
        
        # Crop again to remove black borders
        img = crop_image_from_gray(img)
        
        # Apply enhancement
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX)
        enhanced = cv2.addWeighted(img, 4, blur, -4, 128)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error in circle_crop: {str(e)}")
        # Return resized original image if preprocessing fails
        return cv2.resize(img, (224, 224))

# ------------------ Grad-CAM ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out'):
    """Generate Grad-CAM heatmap for model interpretability"""
    try:
        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_output = predictions[:, pred_index]
        
        # The gradient of the output neuron with regard to the output feature map
        grads = tape.gradient(class_output, conv_outputs)
        
        # A vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), pred_index.numpy()
    except Exception as e:
        logger.error(f"Error in make_gradcam_heatmap: {str(e)}")
        # Return empty heatmap if Grad-CAM fails
        return np.zeros((7, 7)), 0

def overlay_heatmap(img, heatmap, alpha=0.4, threshold=0.2):
    """Overlay heatmap on original image"""
    try:
        # Resize the heatmap to match the size of the original image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Normalize heatmap and convert to color map
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
        # Apply threshold to show only strong activations
        heatmap_resized = np.where(heatmap_resized > threshold * 255, heatmap_resized, 0)
        
        # Apply color map
        color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Overlay the heatmap on the original image
        superimposed_img = cv2.addWeighted(img, 1 - alpha, color_map, alpha, 0)
        
        return superimposed_img
    except Exception as e:
        logger.error(f"Error in overlay_heatmap: {str(e)}")
        return img

# ------------------ Prediction Pipeline ------------------
def preprocess_and_predict(image_path, image_name):
    """Complete pipeline for image preprocessing and prediction"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        preprocessed = circle_crop(img)
        resized = cv2.resize(preprocessed, (224, 224))
        img_array = np.expand_dims(resized / 255.0, axis=0)
        
        # Get model and make prediction
        model = get_model()
        predictions = model.predict(img_array)[0]
        pred_index = np.argmax(predictions)
        
        # Generate Grad-CAM
        heatmap, _ = make_gradcam_heatmap(img_array, model)
        gradcam = overlay_heatmap(resized.copy(), heatmap)
        
        # Ensure media directory exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        # Save processed images
        pre_path = os.path.join(settings.MEDIA_ROOT, f'preprocessed_{image_name}')
        gradcam_path = os.path.join(settings.MEDIA_ROOT, f'gradcam_{image_name}')
        
        # Convert RGB back to BGR for saving
        cv2.imwrite(pre_path, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR))
        
        # Format predictions as dictionary with labels and percentages
        probability_dict = {
            diagnosis_mapping[i]: f"{prob*100:.2f}%" 
            for i, prob in enumerate(predictions)
        }
        
        return (
            diagnosis_mapping[pred_index], 
            f'preprocessed_{image_name}', 
            f'gradcam_{image_name}', 
            probability_dict
        )
        
    except Exception as e:
        logger.error(f"Error in preprocess_and_predict: {str(e)}")
        raise e

# ------------------ Django View ------------------
def detect(request):
    """Main view for diabetic retinopathy detection"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                img_file = request.FILES['image']
                
                # Save uploaded file
                path = default_storage.save('uploaded/' + img_file.name, img_file)
                full_path = os.path.join(settings.MEDIA_ROOT, path)
                
                # Process image and get prediction
                prediction, preprocessed_file, gradcam_file, probabilities = preprocess_and_predict(
                    full_path, img_file.name
                )
                
                context = {
                    'original': path,
                    'preprocessed': preprocessed_file,
                    'gradcam': gradcam_file,
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'MEDIA_URL': settings.MEDIA_URL,
                    'form': form
                }
                
                return render(request, 'detector/index.html', context)
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                form.add_error(None, f"Error processing image: {str(e)}")
        else:
            logger.warning("Form validation failed")
    else:
        form = UploadImageForm()
    
    return render(request, 'detector/index.html', {'form': form})

def health_check(request):
    """Health check endpoint for deployment"""
    try:
        # Try to load model to check if everything is working
        model = get_model()
        return JsonResponse({
            'status': 'healthy',
            'model_loaded': model is not None
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=500)
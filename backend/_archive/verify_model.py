import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Path to the model checkpoint
model_path = r'c:\Users\deonb\OneDrive\Desktop\ESE\backend\model_checkpoint.h5'
test_dir = r'c:\Users\deonb\OneDrive\Desktop\ESE\GARBAGE CLASSIFICATION\TEST'
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

results = []

for cls in classes:
    cls_path = os.path.join(test_dir, cls)
    if not os.path.exists(cls_path):
        print(f"Directory not found: {cls_path}")
        continue
    
    # Get all images in the directory
    files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in {cls_path}")
        continue
    
    # Pick the first image for testing
    test_file = files[0]
    img_path = os.path.join(cls_path, test_file)
    
    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_cls = classes[pred_idx]
    confidence = predictions[0][pred_idx]
    
    results.append({
        'Actual': cls,
        'Predicted': pred_cls,
        'Confidence': f"{confidence:.2%}",
        'Correct': 'YES' if cls == pred_cls else 'NO'
    })

# Display results as a table
df = pd.DataFrame(results)
print("\n--- Test Results ---")
print(df.to_string(index=False))

# Calculate Overall Accuracy on these samples
correct_count = sum(1 for r in results if r['Correct'] == 'YES')
print(f"\nSample Accuracy: {correct_count}/{len(results)} ({correct_count/len(results):.2%})")

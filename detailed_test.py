import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model
model = tf.keras.models.load_model('improved_waste_classifier.h5')
class_names = ['aluminium', 'paper', 'glass', 'plastic', 'cardboard']

# Test the image
img = Image.open('test_image.jpeg').resize((224, 224))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)

print("=== DETAILED PREDICTION RESULTS ===")
print(f"Image: test_image.jpg")
print("\nAll predictions:")

# Show all category predictions
for i, (category, confidence) in enumerate(zip(class_names, predictions[0])):
    marker = " ← WINNER" if i == np.argmax(predictions[0]) else ""
    print(f"{category:>10}: {confidence:.3f} ({confidence*100:.1f}%){marker}")

print(f"\nFinal prediction: {class_names[np.argmax(predictions[0])]}")
print(f"Confidence: {np.max(predictions[0]):.3f}")

# Interpretation
max_conf = np.max(predictions[0])
if max_conf > 0.7:
    print("→ Model is fairly confident")
elif max_conf > 0.5:
    print("→ Model is somewhat sure")  
elif max_conf > 0.3:
    print("→ Model is guessing")
else:
    print("→ Model has no idea")
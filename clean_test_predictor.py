import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def clean_prediction(model_path, image_path):
    """
    Clean prediction for 6 material classes (no trash bias)
    """
    
    # Updated class names - NO TRASH!
    class_names = ['aluminium', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
    
    # Load model
    print("Loading clean material classifier...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and show original image
    print(f"\nProcessing image: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    
    # Show image details
    print(f"Original image size: {original_image.size}")
    
    # Preprocess image (same as training)
    image = original_image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0,1]
    image_array = image_array / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    print("\nMaking clean material prediction...")
    predictions = model.predict(image_batch, verbose=0)
    
    # Show ALL probabilities
    print("\n" + "="*50)
    print("CLEAN MATERIAL CLASSIFICATION:")
    print("="*50)
    
    sorted_indices = np.argsort(predictions[0])[::-1]
    
    for i in sorted_indices:
        class_name = class_names[i]
        probability = predictions[0][i] * 100
        bar = "█" * max(1, int(probability / 3))  # Visual bar
        status = "🎯" if i == sorted_indices[0] else "  "
        print(f"{status} {class_name:>10}: {probability:6.2f}% {bar}")
    
    # Prediction details
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS:")
    print("="*50)
    
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    max_prob = np.max(predictions[0])
    
    # Get top 2 predictions
    top_2_indices = np.argsort(predictions[0])[::-1][:2]
    second_prob = predictions[0][top_2_indices[1]]
    confidence_gap = (max_prob - second_prob) * 100
    
    print(f"🎯 Predicted material: {predicted_class.upper()}")
    print(f"📊 Confidence: {max_prob*100:.2f}%")
    print(f"🥈 Second choice: {class_names[top_2_indices[1]]} ({second_prob*100:.2f}%)")
    print(f"📈 Confidence gap: {confidence_gap:.2f}%")
    
    # Prediction quality assessment
    print(f"\nPrediction Quality:")
    if max_prob > 0.85:
        print("✅ HIGH CONFIDENCE - Very certain prediction")
    elif max_prob > 0.65:
        print("👍 GOOD CONFIDENCE - Solid prediction")
    elif max_prob > 0.45:
        print("⚠️  MODERATE CONFIDENCE - Uncertain prediction")
    else:
        print("❓ LOW CONFIDENCE - Very uncertain prediction")
    
    if confidence_gap > 40:
        print("✅ CLEAR DISTINCTION - Well separated from other classes")
    elif confidence_gap > 20:
        print("👍 GOOD SEPARATION - Reasonably distinct prediction")
    else:
        print("⚠️  CLOSE CALL - Similar probabilities for multiple classes")
    
    # Entropy (uncertainty measure)
    entropy = -np.sum(predictions[0] * np.log(predictions[0] + 1e-8))
    max_entropy = np.log(len(class_names))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    print(f"🔀 Prediction uncertainty: {normalized_entropy:.3f} (0=certain, 1=random)")
    
    # Visualize results
    plt.figure(figsize=(18, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Input Image\n{image_path.split('/')[-1]}", fontsize=12)
    plt.axis('off')
    
    # Preprocessed image (denormalized for display)
    plt.subplot(1, 3, 2)
    display_image = image_array * std + mean
    display_image = np.clip(display_image, 0, 1)
    plt.imshow(display_image)
    plt.title("Preprocessed Image\n(Model Input)", fontsize=12)
    plt.axis('off')
    
    # Probability bar chart with better visualization
    plt.subplot(1, 3, 3)
    probabilities = predictions[0] * 100
    
    # Color coding for better visualization
    colors = []
    for i, prob in enumerate(probabilities):
        if i == predicted_idx:
            colors.append('darkgreen')  # Predicted class
        elif prob > 10:
            colors.append('orange')  # Significant probability
        else:
            colors.append('lightblue')  # Low probability
    
    bars = plt.bar(range(len(class_names)), probabilities, color=colors)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylabel('Probability (%)')
    plt.title(f'Material Classification Results\nPredicted: {predicted_class.upper()} ({max_prob*100:.1f}%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add percentage labels on significant bars
    for bar, prob in zip(bars, probabilities):
        if prob > 8:  # Only show labels for probabilities > 8%
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, max_prob * 100

def test_multiple_clean_images():
    """Test multiple images with clean 6-class model"""
    
    # You'll need to update this path to your new clean model
    MODEL_PATH = input("Enter path to your clean model (.h5 file): ").strip()
    
    if not MODEL_PATH:
        print("No model path provided")
        return
    
    # Test with different images
    test_images = []
    
    print("\nEnter paths to test images (press Enter when done):")
    while True:
        path = input("Image path (or Enter to finish): ").strip()
        if not path:
            break
        test_images.append(path)
    
    if not test_images:
        print("No images provided for testing")
        return
    
    print(f"\nTesting {len(test_images)} images with CLEAN model...")
    print("="*70)
    
    results = []
    class_names = ['aluminium', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n--- TEST IMAGE {i}: {image_path.split('/')[-1]} ---")
        try:
            predicted_class, confidence = clean_prediction(MODEL_PATH, image_path)
            results.append((image_path, predicted_class, confidence))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("CLEAN MODEL PREDICTION SUMMARY:")
    print("="*70)
    
    class_counts = {}
    confidence_sum = 0
    
    for path, pred_class, conf in results:
        filename = path.split('/')[-1]
        print(f"{filename:>25}: {pred_class:>10} ({conf:5.1f}%)")
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        confidence_sum += conf
    
    if results:
        avg_confidence = confidence_sum / len(results)
        print(f"\nAverage confidence: {avg_confidence:.1f}%")
    
    print(f"\nClass distribution in predictions:")
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        percentage = (count / len(results)) * 100 if results else 0
        print(f"  {class_name:>10}: {count:>2} images ({percentage:5.1f}%)")
    
    # Check for bias issues
    if len(class_counts) == 1:
        dominant_class = list(class_counts.keys())[0]
        print(f"\n⚠️  WARNING: All images classified as '{dominant_class}'")
        print("This might indicate a remaining bias issue.")
    elif len(class_counts) >= 4:
        print(f"\n✅ GOOD: Diverse predictions across {len(class_counts)} different classes")
        print("The clean model shows much better class distribution!")
    
    return results

def quick_test_single_image():
    """Quick test for a single image"""
    MODEL_PATH = input("Enter path to your clean model (.h5 file): ").strip()
    IMAGE_PATH = input("Enter image path: ").strip()
    
    if not MODEL_PATH or not IMAGE_PATH:
        print("Model path and image path required")
        return
    
    try:
        predicted_class, confidence = clean_prediction(MODEL_PATH, IMAGE_PATH)
        
        print(f"\n{'='*50}")
        print(f"QUICK RESULT:")
        print(f"Material: {predicted_class.upper()}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"{'='*50}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    print("Clean Material Classification Test Tool")
    print("="*45)
    print("6 Classes: aluminium, cardboard, glass, metal, paper, plastic")
    print("NO TRASH CLASS - Clean predictions only!")
    print("="*45)
    
    choice = input("1. Test single image\n2. Test multiple images\nChoice: ").strip()
    
    if choice == "1":
        quick_test_single_image()
    elif choice == "2":
        test_multiple_clean_images()
    else:
        print("Invalid choice")
        
print("\n🧹 Clean material classification ready!")
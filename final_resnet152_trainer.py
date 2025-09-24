import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("TensorFlow version:", tf.version.VERSION)
print("="*80)
print("UPDATED RESNET152 WASTE CLASSIFICATION TRAINER")
print("6 Classes (No Trash) - Clean Classification!")
print("="*80)

# Configuration optimized for clean 6-class classification
DATASET_PATH = "MultiClassDataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 45
# REMOVED 'trash' class for cleaner classification
CATEGORIES = ['aluminium', 'cardboard', 'glass', 'metal', 'paper', 'plastic']

def load_clean_dataset():
    """Load dataset without the problematic 'trash' class"""
    all_images = []
    all_labels = []
    
    print("Loading CLEAN 6-class dataset (no trash)...")
    print("-" * 60)
    
    for idx, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)
        
        if not os.path.exists(category_path):
            print(f"WARNING: {category} folder not found!")
            continue
            
        # Get all images for this category
        image_files = []
        for file in os.listdir(category_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                image_files.append(os.path.join(category_path, file))
        
        # Use up to 600 images per class for balanced training
        max_per_class = min(600, len(image_files))
        
        if len(image_files) > max_per_class:
            selected_images = random.sample(image_files, max_per_class)
        else:
            selected_images = image_files
        
        all_images.extend(selected_images)
        all_labels.extend([idx] * len(selected_images))
        
        print(f"{category:>12}: {len(selected_images):>4} images (class {idx})")
    
    print("-" * 60)
    print(f"{'TOTAL':>12}: {len(all_images):>4} images selected for training")
    
    return np.array(all_images), np.array(all_labels), CATEGORIES

def enhanced_augmentation(image):
    """Enhanced augmentation for better material recognition"""
    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Enhanced color augmentations for material distinction
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.1)
    
    # Random crop and zoom with more variation
    if tf.random.uniform([]) > 0.2:
        crop_factor = tf.random.uniform([], 0.75, 1.0)
        h, w = int(224 * crop_factor), int(224 * crop_factor)
        image = tf.image.random_crop(image, [h, w, 3])
        image = tf.image.resize(image, IMG_SIZE)
    
    # Random rotation for better generalization
    if tf.random.uniform([]) > 0.4:
        angle = tf.random.uniform([], -0.15, 0.15)  # ~8 degrees
        try:
            image = tf.contrib.image.rotate(image, angle)
        except:
            pass
    
    return image

def preprocess_image(image_path, label, is_training=True):
    """Robust image preprocessing optimized for 6 material classes"""
    try:
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        
        # Resize to target size
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        
        # Apply enhanced augmentation during training
        if is_training:
            image = enhanced_augmentation(image)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image, label
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        black_image = tf.zeros([*IMG_SIZE, 3], dtype=tf.float32)
        return black_image, label

def create_efficient_dataset(image_paths, labels, is_training=True):
    """Create efficient dataset pipeline"""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)
    
    # Parallel processing
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batching and prefetching
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_clean_resnet152_model(num_classes):
    """Create ResNet152 optimized for 6 clean material classes"""
    print(f"Building ResNet152 for {num_classes} CLEAN material classes...")
    
    # Load pre-trained ResNet152
    base_model = tf.keras.applications.ResNet152(
        input_shape=(*IMG_SIZE, 3),
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with optimized architecture for material classification
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    
    # ResNet152 base
    x = base_model(inputs, training=False)
    
    # Enhanced classification head for better material distinction
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Material-specific feature layers
    x = tf.keras.layers.Dense(1024, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.008),
                             name='material_features_1024')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.008),
                             name='material_features_512')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.005),
                             name='material_features_256')(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    
    # Final classification layer for 6 classes
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', 
                                   name='clean_material_classification')(x)
    
    model = tf.keras.Model(inputs, outputs, name='ResNet152_CleanMaterialClassifier')
    
    print(f"Clean model created:")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model, base_model

def plot_detailed_confusion_matrix(y_true, y_pred, class_names, save_path='clean_confusion_matrix.png'):
    """Plot detailed confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Clean Material Classification Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved as: {save_path}")

def plot_training_history(history, save_path='clean_training_history.png'):
    """Plot comprehensive training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], 'o-', label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 's-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Clean Model Accuracy Progress', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], 'o-', label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 's-', label='Validation Loss', linewidth=2)
    ax2.set_title('Clean Model Loss Progress', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-class accuracy breakdown (if available)
    try:
        # Show final epoch accuracy details
        final_acc = history.history['val_accuracy'][-1]
        ax3.bar(range(1), [final_acc], color='green', alpha=0.7)
        ax3.set_title('Final Validation Accuracy', fontsize=14)
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.set_xticks([0])
        ax3.set_xticklabels(['Clean Model'])
        ax3.text(0, final_acc + 0.02, f'{final_acc:.3f}', ha='center', fontsize=12)
    except:
        ax3.text(0.5, 0.5, 'Accuracy Details', ha='center', va='center', transform=ax3.transAxes)
    
    # Training phase visualization
    if len(history.history['val_accuracy']) > 0:
        epochs = range(1, len(history.history['val_accuracy']) + 1)
        ax4.plot(epochs, history.history['val_accuracy'], 'g-', 
                label='Validation Progress', linewidth=2)
        ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% Target')
        ax4.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% Target')
        ax4.set_title('Training Progress Overview', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training history saved as: {save_path}")

def train_clean_resnet152():
    """Main training function for clean 6-class material classification"""
    
    # Load clean dataset (no trash)
    image_paths, labels, class_names = load_clean_dataset()
    
    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\nClass weights for balanced training:")
    for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {class_name}: {weight:.3f}")
    
    # Split dataset with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels,
        test_size=0.25,  # 25% validation for robust evaluation
        random_state=42,
        stratify=labels
    )
    
    print(f"\nClean dataset split:")
    print(f"Training:   {len(X_train):>5} images ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"Validation: {len(X_val):>5} images ({len(X_val)/len(image_paths)*100:.1f}%)")
    
    # Create efficient datasets
    print("\nCreating optimized data pipelines...")
    train_dataset = create_efficient_dataset(X_train, y_train, is_training=True)
    val_dataset = create_efficient_dataset(X_val, y_val, is_training=False)
    
    # Create clean model
    model, base_model = create_clean_resnet152_model(len(class_names))
    
    print(f"\nClean training classes: {', '.join(class_names)}")
    
    # STAGE 1: Train classifier head (ResNet152 frozen)
    print("\n" + "="*80)
    print("STAGE 1: Training CLEAN classification head (ResNet152 backbone frozen)")
    print("="*80)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    stage1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'clean_resnet152_stage1_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("Starting Stage 1 training (clean classes)...")
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=22,
        class_weight=class_weight_dict,
        callbacks=stage1_callbacks,
        verbose=1
    )
    
    stage1_acc = max(history1.history['val_accuracy'])
    print(f"\nStage 1 Clean Results:")
    print(f"Best validation accuracy: {stage1_acc:.4f} ({stage1_acc*100:.1f}%)")
    
    # STAGE 2: Fine-tune ResNet152 layers
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning ResNet152 for material distinction")
    print("="*80)
    
    # Unfreeze ResNet152 for fine-tuning
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    fine_tune_at = len(base_model.layers) - 35  # Unfreeze last 35 layers
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    print(f"Trainable layers: {trainable_count}")
    print(f"Fine-tuning from layer: {fine_tune_at}")
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    stage2_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=6,
            min_lr=1e-8,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'clean_resnet152_stage2_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("Starting Stage 2 clean fine-tuning...")
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=28,
        class_weight=class_weight_dict,
        callbacks=stage2_callbacks,
        verbose=1
    )
    
    # Combine training histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL CLEAN MODEL EVALUATION")
    print("="*80)
    
    final_loss, final_accuracy = model.evaluate(val_dataset, verbose=1)
    
    print(f"\nFinal Clean Results:")
    print(f"Validation Loss:     {final_loss:.4f}")
    print(f"Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")
    
    # Detailed predictions analysis
    print("\nGenerating detailed predictions...")
    all_predictions = []
    all_true_labels = []
    
    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        all_predictions.extend(np.argmax(preds, axis=1))
        all_true_labels.extend(labels.numpy())
    
    # Classification report
    print("\nDetailed Clean Classification Report:")
    print("="*70)
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=class_names, digits=4))
    
    # Plot confusion matrix
    plot_detailed_confusion_matrix(all_true_labels, all_predictions, class_names)
    
    # Sample predictions with better distribution check
    print(f"\nSample Predictions (first 30):")
    print("-" * 60)
    class_prediction_counts = {name: 0 for name in class_names}
    correct_count = 0
    
    for i in range(min(30, len(all_predictions))):
        pred_class = class_names[all_predictions[i]]
        true_class = class_names[all_true_labels[i]]
        correct = "✓" if all_predictions[i] == all_true_labels[i] else "✗"
        if correct == "✓":
            correct_count += 1
        class_prediction_counts[pred_class] += 1
        print(f"{correct} True: {true_class:>10} | Pred: {pred_class:>10}")
    
    sample_accuracy = correct_count / min(30, len(all_predictions))
    print(f"\nSample accuracy: {sample_accuracy:.3f} ({sample_accuracy*100:.1f}%)")
    
    print(f"\nPrediction distribution in sample:")
    for class_name, count in class_prediction_counts.items():
        print(f"  {class_name}: {count} predictions")
    
    # Plot training history
    class MockHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    plot_training_history(MockHistory(combined_history))
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_model_name = f'ResNet152_CleanMaterialClassifier_{timestamp}.h5'
    model.save(final_model_name)
    
    print(f"\n✅ Clean model saved as: {final_model_name}")
    
    # Performance assessment
    print("\n" + "="*80)
    print("CLEAN MODEL ASSESSMENT")
    print("="*80)
    
    if final_accuracy > 0.88:
        print("🏆 OUTSTANDING! Your clean ResNet152 model achieved excellent performance!")
        print("   This is production-ready for clean material classification.")
    elif final_accuracy > 0.78:
        print("🎉 EXCELLENT! Your clean ResNet152 model performs very well!")
        print("   Great results for material classification without trash bias.")
    elif final_accuracy > 0.68:
        print("👍 GOOD! Your clean ResNet152 model shows solid performance.")
        print("   Much better classification without the problematic trash class.")
    elif final_accuracy > 0.55:
        print("📈 DECENT! Clean model is learning well, improvement over biased version.")
    else:
        print("📊 NEEDS IMPROVEMENT! But should be much better than trash-biased model.")
    
    print(f"\nClass distribution check:")
    unique_preds, counts = np.unique(all_predictions, return_counts=True)
    for pred_idx, count in zip(unique_preds, counts):
        percentage = (count / len(all_predictions)) * 100
        print(f"  {class_names[pred_idx]}: {count} predictions ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("CLEAN TRAINING COMPLETE!")
    print("="*80)
    
    return model, combined_history, class_names, final_accuracy

if __name__ == "__main__":
    try:
        # Enable memory growth for GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} GPU(s) configured")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected - using CPU (training will be slower)")
        
        # Run clean training
        model, history, classes, accuracy = train_clean_resnet152()
        
        print(f"\n🎯 FINAL CLEAN RESULT: {accuracy*100:.1f}% accuracy achieved!")
        print(f"📁 Clean model saved for deployment")
        print(f"📊 Training history and confusion matrix generated")
        print(f"🧹 No more trash bias - clean 6-class classification!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
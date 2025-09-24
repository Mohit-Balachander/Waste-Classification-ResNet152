# dataset_analyzer.py
import os

def analyze_dataset():
    """Analyze the MultiClassDataset folder structure and count images"""
    
    dataset_path = "MultiClassDataset"  # Note: Capital M as per your structure
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} folder not found!")
        print("Please check if the folder name is correct.")
        return
    
    categories = ['aluminium', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    total_images = 0
    category_counts = {}
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        
        if os.path.exists(category_path):
            # Count image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(image_extensions)]
            
            count = len(image_files)
            category_counts[category] = count
            total_images += count
            
            print(f"{category:>12}: {count:>4} images")
            
            # Show first few filenames as examples
            if count > 0:
                examples = image_files[:3]
                print(f"{'':>12}   Examples: {', '.join(examples)}")
        else:
            category_counts[category] = 0
            print(f"{category:>12}: {0:>4} images (FOLDER MISSING)")
    
    print("-" * 60)
    print(f"{'TOTAL':>12}: {total_images:>4} images")
    
    # Analysis
    print("\n" + "="*60)
    print("DATASET ASSESSMENT")
    print("="*60)
    
    if total_images == 0:
        print("❌ NO IMAGES FOUND!")
        print("Please check your folder structure and image files.")
        return
    
    # Find min/max/avg
    non_zero_counts = [count for count in category_counts.values() if count > 0]
    
    if non_zero_counts:
        min_images = min(non_zero_counts)
        max_images = max(non_zero_counts)
        avg_images = sum(non_zero_counts) / len(non_zero_counts)
        
        print(f"Minimum images per class: {min_images}")
        print(f"Maximum images per class: {max_images}")
        print(f"Average images per class: {avg_images:.1f}")
        
        # Data balance assessment
        balance_ratio = min_images / max_images if max_images > 0 else 0
        print(f"Balance ratio (min/max): {balance_ratio:.2f}")
        
        if balance_ratio < 0.5:
            print("⚠️  HIGHLY IMBALANCED dataset")
        elif balance_ratio < 0.8:
            print("📊 MODERATELY IMBALANCED dataset")
        else:
            print("✅ WELL BALANCED dataset")
    
    # Training recommendations
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATIONS")
    print("="*60)
    
    if total_images < 700:  # Less than 100 per class
        print("❌ INSUFFICIENT DATA for reliable ResNet152 training")
        print("   Recommendations:")
        print("   • Add more images (aim for 300+ per class)")
        print("   • Use smaller model (MobileNet/ResNet50)")
        print("   • Apply heavy data augmentation")
        
    elif total_images < 1400:  # 100-200 per class
        print("⚠️  MARGINAL DATA for ResNet152")
        print("   Recommendations:")
        print("   • ResNet152 may overfit")
        print("   • Consider ResNet50 or EfficientNet")
        print("   • Use strong regularization")
        
    elif total_images < 2100:  # 200-300 per class
        print("📈 ACCEPTABLE DATA for careful ResNet152 training")
        print("   Recommendations:")
        print("   • Use progressive training stages")
        print("   • Apply moderate augmentation")
        print("   • Monitor for overfitting")
        
    else:  # 300+ per class
        print("✅ GOOD DATA for ResNet152 training")
        print("   Recommendations:")
        print("   • ResNet152 should work well")
        print("   • Can use standard training approach")
        
    # Missing classes warning
    missing_classes = [cat for cat, count in category_counts.items() if count == 0]
    if missing_classes:
        print(f"\n❌ MISSING CLASSES: {', '.join(missing_classes)}")
        print("   Either add images for these classes or remove them from training")
    
    return category_counts, total_images

if __name__ == "__main__":
    try:
        analyze_dataset()
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
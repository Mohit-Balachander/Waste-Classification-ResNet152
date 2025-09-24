import os

# Check your dataset structure
dataset_path = 'multiclassdataset'

print("=== DATASET OVERVIEW ===")
print("-" * 40)

total_images = 0
categories_found = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        # Count image files only
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        count = len(image_files)
        
        categories_found.append(category)
        total_images += count
        print(f"{category:>12}: {count:>4} images")

print("-" * 40)
print(f"{'Total':>12}: {total_images:>4} images")
print(f"Categories: {len(categories_found)}")
print(f"Found: {', '.join(sorted(categories_found))}")

# Check if any category has very few images
print("\n=== ANALYSIS ===")
low_data_categories = []
for category in categories_found:
    category_path = os.path.join(dataset_path, category)
    image_files = [f for f in os.listdir(category_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    count = len(image_files)
    if count < 100:
        low_data_categories.append(f"{category} ({count})")

if low_data_categories:
    print("⚠️  Categories with few images:")
    for cat in low_data_categories:
        print(f"   - {cat}")
else:
    print("✅ All categories have good amount of data")

print(f"\nAverage per category: {total_images // len(categories_found) if categories_found else 0} images")
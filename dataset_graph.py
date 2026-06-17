import os
import matplotlib.pyplot as plt

# Dataset yolu
dataset_path = "dataset/train"

# Sınıflar ve görüntü sayıları
classes = []
counts = []

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        image_count = len(os.listdir(class_path))

        classes.append(class_name)

        counts.append(image_count)

# Grafik boyutu
plt.figure(figsize=(12, 6))

# Bar chart
plt.bar(classes, counts)

# Başlıklar
plt.title("Flower Dataset Class Distribution")

plt.xlabel("Flower Classes")

plt.ylabel("Number of Images")

# Yazıları döndür
plt.xticks(rotation=45)

# Layout düzelt
plt.tight_layout()

# Kaydet
plt.savefig("dataset_distribution.png")

# Göster
plt.show()
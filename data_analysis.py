import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Set style untuk plot yang lebih menarik
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🌸 ANALISIS LENGKAP DATASET IRIS 🌸")
print("=" * 60)

# 1. LOAD DATA
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
df['target'] = iris.target

print("📊 INFORMASI DATASET:")
print(f"   Jumlah sampel: {len(df)}")
print(f"   Jumlah fitur: {len(iris.feature_names)}")
print(f"   Jumlah kelas: {len(iris.target_names)}")
print(f"   Kelas: {', '.join(iris.target_names)}")
print()

# 2. STATISTIK DASAR
print("📈 STATISTIK DASAR:")
print(df.describe().round(2))
print()

# 3. DISTRIBUSI KELAS
print("🎯 DISTRIBUSI KELAS:")
class_counts = df['species'].value_counts()
for species, count in class_counts.items():
    print(f"   {species}: {count} sampel")
print()

# 4. VISUALISASI LENGKAP
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ANALISIS VISUAL DATASET IRIS', fontsize=16, fontweight='bold')

# Plot 1: Scatter plot utama
axes[0,0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], 
                 c=df['target'], cmap='viridis', alpha=0.7)
axes[0,0].set_xlabel('Sepal Length (cm)')
axes[0,0].set_ylabel('Sepal Width (cm)')
axes[0,0].set_title('Sepal: Length vs Width')

# Plot 2: Petal scatter
axes[0,1].scatter(df['petal length (cm)'], df['petal width (cm)'], 
                 c=df['target'], cmap='viridis', alpha=0.7)
axes[0,1].set_xlabel('Petal Length (cm)')
axes[0,1].set_ylabel('Petal Width (cm)')
axes[0,1].set_title('Petal: Length vs Width')

# Plot 3: Box plot sepal length
df.boxplot(column='sepal length (cm)', by='species', ax=axes[1,0])
axes[1,0].set_title('Distribusi Sepal Length per Spesies')
axes[1,0].set_xlabel('Species')

# Plot 4: Histogram
axes[1,1].hist([df[df['species']=='setosa']['petal length (cm)'],
               df[df['species']=='versicolor']['petal length (cm)'],
               df[df['species']=='virginica']['petal length (cm)']], 
              bins=15, alpha=0.7, label=iris.target_names)
axes[1,1].set_xlabel('Petal Length (cm)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribusi Petal Length')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# 5. CORRELATION HEATMAP
plt.figure(figsize=(10, 6))
correlation_matrix = df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('KORELASI ANTAR FITUR IRIS DATASET', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 6. MACHINE LEARNING
print("🤖 MACHINE LEARNING CLASSIFICATION:")
print("-" * 40)

# Persiapan data
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred = rf_model.predict(X_test_scaled)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model: {accuracy:.2%}")
print()
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("🔍 TINGKAT KEPENTINGAN FITUR:")
for _, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")
print()

# 7. PREDIKSI CONTOH BARU
print("🔮 PREDIKSI CONTOH BARU:")
print("-" * 30)

# Contoh data baru
sample_data = np.array([[5.1, 3.5, 1.4, 0.2],  # Mirip setosa
                       [6.2, 2.8, 4.8, 1.8],   # Mirip virginica
                       [5.7, 2.8, 4.5, 1.3]])  # Mirip versicolor

sample_scaled = scaler.transform(sample_data)
predictions = rf_model.predict(sample_scaled)
probabilities = rf_model.predict_proba(sample_scaled)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    species = iris.target_names[pred]
    confidence = max(prob) * 100
    print(f"   Sampel {i+1}: {species} (confidence: {confidence:.1f}%)")

print()
print("✨ ANALISIS SELESAI! ✨")
print("=" * 60)
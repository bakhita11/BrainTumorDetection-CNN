# Multimodal CNN for Brain MRI (T1, T2, FLAIR Fusion (Results and Visualizations)

Full Code:
# Multimodal CNN (T1, T2, FLAIR)
# Patient-level split (70/15/15)

import os, h5py, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1) Loader (patient-aware, fusion)
def load_multimodal_by_patient(csv_file, base_path, max_patients=None, slices_per_patient=None, resize=(128,128)):
    df = pd.read_csv(csv_file)
    df['slice_path'] = df['slice_path'].apply(lambda x: os.path.join(base_path, os.path.basename(x)))

    grouped = df.groupby('volume')
    patient_ids = list(grouped.groups.keys())

    if max_patients:
        patient_ids = patient_ids[:max_patients]

    X, y, patient_labels = [], [], []

    for pid in patient_ids:
        patient_df = grouped.get_group(pid)
        patient_df = patient_df.sample(frac=1, random_state=42).reset_index(drop=True)

        if slices_per_patient:
            patient_df = patient_df.iloc[:slices_per_patient]

        for _, row in patient_df.iterrows():
            if not os.path.exists(row['slice_path']):
                continue
            with h5py.File(row['slice_path'], 'r') as f:
                img = f['image'][:, :, :3].astype(np.float32)   # fusion: T1, T2, FLAIR
                img = cv2.resize(img, resize)

                # Per-channel min-max normalization
                for c in range(img.shape[-1]):
                    ch = img[:,:,c]
                    mn, mx = float(ch.min()), float(ch.max())
                    img[:,:,c] = (ch - mn) / (mx - mn + 1e-8)

                X.append(img)
                y.append(int(row['target']))
                patient_labels.append(pid)

    return np.array(X, dtype='float32'), np.array(y, dtype='int'), np.array(patient_labels)

# 2) Load data

csv_path = "/content/brats2020/BraTS20 Training Metadata.csv"
base_path = "/content/brats2020/BraTS2020_training_data/content/data/"

# Limit to avoid RAM crash (250 patients)
X, y, patient_ids = load_multimodal_by_patient(
    csv_path, base_path, max_patients=250, slices_per_patient=50, resize=(128,128)
)

print("Data shape:", X.shape, y.shape)
print("Unique patients:", len(np.unique(patient_ids)))

# 3) Patient-level split (70/15/15)
unique_pids = np.unique(patient_ids)

# hold out 15% for test
trainval_pids, test_pids = train_test_split(unique_pids, test_size=0.15, random_state=42)

# Then: from remaining 85%, carve out ~17.6% for validation (~15% of total)
val_size = 0.1764705882
train_pids, val_pids = train_test_split(trainval_pids, test_size=val_size, random_state=42)

def mask_by_pids(X, y, pids, patient_ids):
    mask = np.isin(patient_ids, pids)
    return X[mask], y[mask]

X_train, y_train = mask_by_pids(X, y, train_pids, patient_ids)
X_val, y_val     = mask_by_pids(X, y, val_pids, patient_ids)
X_test, y_test   = mask_by_pids(X, y, test_pids, patient_ids)

print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

# 4) Class weights
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))
print("Class weights:", class_weight)

# 5) Model

input_shape = X.shape[1:]
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')   # binary sigmoid insetad of softmax #since it classifies yes tumor or no tumor, no multiclassification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6) Data augmentation (for overfitting.)
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=False  # MRI is not usually flipped vertically
)

val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=32)
val_gen   = val_datagen.flow(X_val, y_val, batch_size=32)

# 7) Training (early stoppin to)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
chk = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    class_weight=class_weight,
    callbacks=[early_stop, chk],
    verbose=1
)

# 8) Evaluation
y_test_prob = model.predict(X_test).ravel()
y_test_pred = (y_test_prob >= 0.5).astype(int)

test_acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tumor','Tumor'], yticklabels=['No Tumor','Tumor'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_aug.png", bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig("roc_curve_aug.png", bbox_inches='tight')
plt.show()

# 9) Training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.savefig("training_val_curves_aug.png", bbox_inches='tight')
plt.show()

# 10) Visualize modalities

def show_modalities(X, y, num_samples=3):
    idxs = np.random.choice(len(X), num_samples, replace=False)
    for idx in idxs:
        img = X[idx]
        lab = y[idx]
        plt.figure(figsize=(12,3))
        plt.subplot(1,4,1); plt.imshow(img[:,:,0], cmap='gray'); plt.title('T1'); plt.axis('off')
        plt.subplot(1,4,2); plt.imshow(img[:,:,1], cmap='gray'); plt.title('T2'); plt.axis('off')
        plt.subplot(1,4,3); plt.imshow(img[:,:,2], cmap='gray'); plt.title('FLAIR'); plt.axis('off')
        plt.subplot(1,4,4); plt.imshow(img); plt.title(f'Fused (label={lab})'); plt.axis('off')
        plt.show()

show_modalities(X, y, num_samples=3)

# Visualization: modalities + fused + predictions

def visualize_prediction(X, y_true, y_prob, idx, class_names={0:"No Tumor", 1:"Tumor"}):
    """
    Show T1, T2, FLAIR, and fused image with prediction info.

    Parameters:
    - X: array of shape (N,H,W,3)
    - y_true: true labels (0/1)
    - y_prob: predicted probabilities (floats [0,1])
    - idx: sample index
    """
    img = X[idx]
    true = y_true[idx]
    prob = y_prob[idx]
    pred = int(prob >= 0.5)

    plt.figure(figsize=(14,4))

    # T1
    plt.subplot(1,4,1)
    plt.imshow(img[:,:,0], cmap="gray")
    plt.title("T1")
    plt.axis("off")

    # T2
    plt.subplot(1,4,2)
    plt.imshow(img[:,:,1], cmap="gray")
    plt.title("T2")
    plt.axis("off")

    # FLAIR
    plt.subplot(1,4,3)
    plt.imshow(img[:,:,2], cmap="gray")
    plt.title("FLAIR")
    plt.axis("off")

    # Fused RGB
    plt.subplot(1,4,4)
    plt.imshow(img)
    plt.title(f"Fused\nTrue: {class_names[true]}\nPred: {class_names[pred]} ({prob:.2f})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example: visualize 4 random test samples
idxs = np.random.choice(len(X_test), 4, replace=False)
for idx in idxs:
    visualize_prediction(X_test, y_test, y_test_prob, idx)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# Compute metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Print metrics
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Full classification report
class_report = classification_report(y_test, y_test_pred, target_names=['No Tumor', 'Tumor'])
print("\nClassification Report:\n")
print(class_report)

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Precision: {precision:.3f}\n")
    f.write(f"Recall: {recall:.3f}\n")
    f.write(f"F1 Score: {f1:.3f}\n\n")
    f.write("Full Classification Report:\n")
    f.write(class_report)

print("\nClassification report saved as 'classification_report.txt'")



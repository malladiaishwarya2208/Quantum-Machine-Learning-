# Quantum Kernel-Enhanced Hybrid Models for Start-Up Success Prediction (Improved Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pennylane as qml
from pennylane.qnn import TorchLayer

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load and preprocess data
file_path = r"C:\\Users\\gowth\\Downloads\\QML Startup Success prediction\\investments_VC.csv"
df = pd.read_csv(file_path, encoding='latin1')
df.columns = df.columns.str.strip()

# Basic cleaning
df = df.dropna(subset=['status'])
df = df[df['status'].isin(['closed', 'acquired'])]
df['label'] = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)

# Feature Selection (Expanded)
selected_features = ['funding_rounds', 'funding_total_usd', 'founded_at',
                     'seed', 'venture', 'equity_crowdfunding', 'angel', 'private_equity',
                     'round_A', 'round_B', 'round_C']
df = df[selected_features + ['label']]

# Date handling
df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['founded_year'] = df['founded_at'].dt.year

df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(',', '')
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')

# Fill missing values with median
for col in ['funding_rounds', 'funding_total_usd', 'founded_year',
            'seed', 'venture', 'equity_crowdfunding', 'angel', 'private_equity',
            'round_A', 'round_B', 'round_C']:
    df[col] = df[col].fillna(df[col].median())

X = df[['funding_rounds', 'funding_total_usd', 'founded_year',
        'seed', 'venture', 'equity_crowdfunding', 'angel', 'private_equity',
        'round_A', 'round_B', 'round_C']]
y = df['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Classical Models for Baseline
print("\n=== Classical Models ===")
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM (Linear Kernel)': SVC(kernel='linear')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# 3. Hybrid Quantum-Classical Model (Quantum-CNN)
print("\n=== Hybrid Quantum-Classical CNN ===")

n_qubits = 4
qlayer_dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(qlayer_dev, interface="torch")
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (6, n_qubits)}
qlayer = TorchLayer(qnode, weight_shapes)

class QuantumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X_train.shape[1], n_qubits)
        self.quantum = qlayer
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = QuantumCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005)

n_epochs = 50
loss_values = []
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_epoch = total_loss / len(train_loader)
    loss_values.append(loss_epoch)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss_epoch:.4f}")

# Plot Loss Curve
plt.plot(range(1, n_epochs+1), loss_values)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Model Evaluation
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_test_tensor)
    preds = torch.argmax(preds, dim=1).cpu().numpy()

print("\nHybrid Model Classification Report:")
print(classification_report(y_test, preds))

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Hybrid Quantum-CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

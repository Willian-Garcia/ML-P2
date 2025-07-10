import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rasterio
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Configuração do dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {DEVICE}")

# Diretórios
base_path = "./new_images"
output_model_path = "./modelo"
report_path = os.path.join(output_model_path, "relatorios", "cnn_pytorch")
os.makedirs(report_path, exist_ok=True)

# Funções utilitárias
def load_band(filename):
    path = os.path.join(base_path, filename + ".tif")
    with rasterio.open(path) as src:
        return src.read(1)

def normalize_band(band, max_val=8000):
    return np.where(band > max_val, 1.0, band / max_val)

# Carregar e preparar dados
print("\n[INFO] Carregando bandas...")
B02 = normalize_band(load_band("new_cut_B02"))
B03 = normalize_band(load_band("new_cut_B03"))
B04 = normalize_band(load_band("new_cut_B04"))
SCL = load_band("new_cut_SCL")

print("[INFO] Pré-processando dados...")
mask = np.isin(SCL, [4, 5, 6])
X = np.stack([B04[mask], B03[mask], B02[mask]], axis=1)
y_raw = SCL[mask]
class_map = {4: 0, 5: 1, 6: 2}
y = np.array([class_map[val] for val in y_raw])

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

# Adicionar dimensão de canal e converter para tensores
X_train_t = torch.tensor(X_train[:, :, None, None], dtype=torch.float32).permute(0, 2, 3, 1).permute(0, 3, 1, 2)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val[:, :, None, None], dtype=torch.float32).permute(0, 2, 3, 1).permute(0, 3, 1, 2)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test[:, :, None, None], dtype=torch.float32).permute(0, 2, 3, 1).permute(0, 3, 1, 2)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Datasets e DataLoaders
batch_size = 2048
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

# Calcular pesos das classes
weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# Modelo CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# Treinamento
epochs = 50
patience = 7
best_val_acc = 0
counter = 0
train_loss_history = []
val_acc_history = []
best_model_state = None

print("[INFO] Treinando modelo CNN com PyTorch...")
start_time = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss_history.append(total_loss / len(train_loader))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(X_batch)
            pred = torch.argmax(output, dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
    acc = correct / total
    val_acc_history.append(acc)

    print(f"[E{epoch+1:02d}] Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_val_acc:
        best_val_acc = acc
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print("[INFO] Early stopping ativado!")
            break
    scheduler.step()

end_time = time.time()
print(f"[INFO] CNN concluído em {end_time - start_time:.2f} segundos.")

# Avaliação
model.load_state_dict(best_model_state)
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

acc_test = accuracy_score(y_true, y_pred)
report_test = classification_report(y_true, y_pred, target_names=["Vegetação", "Não Vegetação", "Água"])

# Validação para relatório
y_val_true, y_val_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        preds = torch.argmax(model(X_batch), dim=1)
        y_val_true.extend(y_batch.numpy())
        y_val_pred.extend(preds.cpu().numpy())
report_val = classification_report(y_val_true, y_val_pred, target_names=["Vegetação", "Não Vegetação", "Água"])

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Vegetação", "Não Vegetação", "Água"]).plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - CNN (PyTorch)")
plt.savefig(os.path.join(report_path, "confusion_matrix_teste.png"))
plt.close()

# Curva de aprendizado
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Loss", color="purple")
plt.plot(val_acc_history, label="Val Acc", color="green")
plt.xlabel("Épocas")
plt.title("Curva de Aprendizado - CNN com PyTorch")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_path, "learning_curve.png"))
plt.close()

# Salvar modelo e relatório
torch.save(best_model_state, os.path.join(output_model_path, "cnn_torch.pt"))

summary_path = os.path.join(report_path, "resumo.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("==== RELATÓRIO DE TREINAMENTO - CNN com PyTorch ====\n\n")
    f.write(f"Dispositivo utilizado: {DEVICE}\n")
    f.write(f"Tempo de treinamento: {end_time - start_time:.2f} segundos\n")
    f.write(f"Épocas: {epoch+1}\n")
    f.write(f"Melhor acurácia de validação: {best_val_acc:.4f}\n")
    f.write(f"Acurácia Teste: {acc_test:.4f}\n\n")
    f.write("==== Classificação - Validação ====\n")
    f.write(report_val + "\n")
    f.write("==== Classificação - Teste ====\n")
    f.write(report_test + "\n")

print("[INFO] Relatório salvo e treinamento finalizado com sucesso.")

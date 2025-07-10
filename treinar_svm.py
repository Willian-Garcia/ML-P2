import os
import time
import numpy as np
import rasterio
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import learning_curve

# Diretórios
base_path = "./new_images"
modelo_path = "./modelo"
report_path = os.path.join(modelo_path, "relatorios", "svm_rbf")
os.makedirs(report_path, exist_ok=True)

def load_band(nome):
    path = os.path.join(base_path, nome + ".tif")
    with rasterio.open(path) as src:
        return src.read(1)

def normalize_band(banda, max_val=8000):
    return np.where(banda > max_val, 1.0, banda / max_val)

print("[INFO] Carregando e normalizando bandas...")
B02 = normalize_band(load_band("new_cut_B02"))
B03 = normalize_band(load_band("new_cut_B03"))
B04 = normalize_band(load_band("new_cut_B04"))
SCL = load_band("new_cut_SCL")

# Máscara das classes desejadas
mask_full = np.isin(SCL, [4, 5, 6])
h, w = SCL.shape

print("[INFO] Iniciando varredura com amostragem balanceada...")
X_total = []
y_total = []
bloco_linhas = 100
samples_per_class = 30000
class_map = {4: 0, 5: 1, 6: 2}
class_counts = {0: 0, 1: 0, 2: 0}

for i in range(0, h, bloco_linhas):
    i_fim = min(i + bloco_linhas, h)
    bloco_mask = mask_full[i:i_fim, :]
    if not np.any(bloco_mask):
        continue

    bloco_B02 = B02[i:i_fim, :][bloco_mask]
    bloco_B03 = B03[i:i_fim, :][bloco_mask]
    bloco_B04 = B04[i:i_fim, :][bloco_mask]
    bloco_SCL = SCL[i:i_fim, :][bloco_mask]
    bloco_X = np.stack([bloco_B04, bloco_B03, bloco_B02], axis=1)
    bloco_y = np.array([class_map[val] for val in bloco_SCL])

    for cls in [0, 1, 2]:
        cls_mask = bloco_y == cls
        if np.sum(cls_mask) == 0 or class_counts[cls] >= samples_per_class:
            continue

        n_rest = samples_per_class - class_counts[cls]
        amostrados_X, amostrados_y = resample(
            bloco_X[cls_mask], bloco_y[cls_mask],
            replace=False, n_samples=min(n_rest, np.sum(cls_mask)),
            random_state=42
        )
        X_total.append(amostrados_X)
        y_total.append(amostrados_y)
        class_counts[cls] += len(amostrados_y)

    if all([class_counts[c] >= samples_per_class for c in [0, 1, 2]]):
        break

X = np.concatenate(X_total)
y = np.concatenate(y_total)

print(f"[INFO] Amostragem final: Vegetação={class_counts[0]}, Não Vegetação={class_counts[1]}, Água={class_counts[2]}")

# Treinamento do modelo
print("[INFO] Treinando SVM com kernel RBF...")
start = time.time()
svm = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
svm.fit(X, y)
end = time.time()

# Salvando modelo
joblib.dump(svm, os.path.join(modelo_path, "svm_rbf_model.pkl"))

# Avaliação
y_pred = svm.predict(X)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, target_names=["Vegetação", "Não Vegetação", "Água"])

# Matriz de confusão
cm_disp = ConfusionMatrixDisplay.from_predictions(
    y, y_pred,
    display_labels=["Vegetação", "Não Vegetação", "Água"],
    cmap=plt.cm.Blues
)
plt.title("Matriz de Confusão - SVM RBF (Amostrada)")
plt.savefig(os.path.join(report_path, "confusion_matrix.png"), bbox_inches='tight')
plt.close()

# Curva de aprendizado com subconjunto pequeno para agilidade
print("[INFO] Gerando curva de aprendizado...")
train_sizes, train_scores, val_scores = learning_curve(
    svm, X, y,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, label="Treino", marker='o')
plt.plot(train_sizes, val_mean, label="Validação", marker='s')
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("Acurácia")
plt.title("Curva de Aprendizado - SVM RBF (Amostrada)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(report_path, "learning_curve.png"))
plt.close()

# Relatório final
with open(os.path.join(report_path, "resumo.txt"), "w", encoding="utf-8") as f:
    f.write("==== RELATÓRIO SVM RBF COM VARREDURA ====\n")
    f.write(f"Tempo de execução: {end - start:.2f} segundos\n")
    f.write(f"Acurácia total: {acc:.4f}\n\n")
    f.write(report)

print("[INFO] Treinamento e relatório finalizados.")

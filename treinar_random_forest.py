# treinar_random_forest.py (versão otimizada)
import os
import time
import numpy as np
import rasterio
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample

print("[INFO] Treinando modelo Random Forest...")

# Caminhos
base_path = "./new_images"
output_path = "./modelo"
report_path = os.path.join(output_path, "relatorios", "random_forest")
os.makedirs(report_path, exist_ok=True)

def load_band(filename):
    with rasterio.open(os.path.join(base_path, filename + ".tif")) as src:
        return src.read(1)

def normalize(band):
    return np.where(band > 8000, 1.0, band / 8000)

print("[INFO] Carregando e processando bandas com varredura...")

# Carregar dados
B02 = normalize(load_band("new_cut_B02"))
B03 = normalize(load_band("new_cut_B03"))
B04 = normalize(load_band("new_cut_B04"))
SCL = load_band("new_cut_SCL")

mask_full = np.isin(SCL, [4, 5, 6])
h, w = SCL.shape

X_total, y_total = [], []
bloco_linhas = 100

for i in range(0, h, bloco_linhas):
    i_fim = min(i + bloco_linhas, h)
    bloco_mask = mask_full[i:i_fim, :]
    if not np.any(bloco_mask):
        continue
    bloco_B04 = B04[i:i_fim, :][bloco_mask]
    bloco_B03 = B03[i:i_fim, :][bloco_mask]
    bloco_B02 = B02[i:i_fim, :][bloco_mask]
    bloco_SCL = SCL[i:i_fim, :][bloco_mask]

    X_bloco = np.stack([bloco_B04, bloco_B03, bloco_B02], axis=1)
    y_bloco = np.array([{4: 0, 5: 1, 6: 2}[val] for val in bloco_SCL])

    X_total.append(X_bloco)
    y_total.append(y_bloco)

X = np.concatenate(X_total, axis=0)
y = np.concatenate(y_total, axis=0)

print(f"[INFO] Total de pixels amostrados: {len(y)}")

# Treinamento do modelo
print("[INFO] Treinando Random Forest...")
start = time.time()
rf = RandomForestClassifier(max_depth=15, n_estimators=100, random_state=42, n_jobs=8, verbose=1)
rf.fit(X, y)
end = time.time()

# Salvando modelo
joblib.dump(rf, os.path.join(output_path, "random_forest.pkl"))

# Avaliação
y_pred = rf.predict(X)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, target_names=["Vegetação", "Não Vegetação", "Água"])

print(f"[INFO] Acurácia total: {acc:.4f}")

# Matriz de confusão
cm_disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Vegetação", "Não Vegetação", "Água"], cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - RF (Varredura)")
plt.savefig(os.path.join(report_path, "confusion_matrix.png"), bbox_inches='tight')
plt.close()

# Importância das features
plt.barh(["B04", "B03", "B02"], rf.feature_importances_, color="forestgreen")
plt.xlabel("Importância")
plt.title("Importância das Bandas - RF")
plt.tight_layout()
plt.savefig(os.path.join(report_path, "feature_importance.png"))
plt.close()

# Curva de aprendizado otimizada
print("[INFO] Gerando curva de aprendizado com amostragem...")

# Amostrar até 500.000 pontos
X_reduzido, y_reduzido = resample(X, y, n_samples=min(500_000, len(y)), random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    rf, X_reduzido, y_reduzido,
    cv=3,
    scoring='accuracy',
    train_sizes=np.linspace(0.2, 1.0, 3),
    n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, label="Treino", marker='o')
plt.plot(train_sizes, val_mean, label="Validação", marker='s')
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("Acurácia")
plt.title("Curva de Aprendizado - RF (Amostrada)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(report_path, "learning_curve.png"))
plt.close()

# Salvar resumo
with open(os.path.join(report_path, "resumo.txt"), "w", encoding="utf-8") as f:
    f.write("==== RELATÓRIO RF COM VARREDURA ====\n")
    f.write(f"Tempo de execução: {end - start:.2f} segundos\n")
    f.write(f"Acurácia total: {acc:.4f}\n\n")
    f.write(report)

print("[INFO] Treinamento e relatórios concluídos.")

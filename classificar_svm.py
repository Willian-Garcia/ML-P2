import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import time

# Diretórios
base_path = "./imagens"
modelo_path = "./modelo"
output_img_path = "./imagens_classificadas"
os.makedirs(output_img_path, exist_ok=True)

# === Carregamento e normalização das bandas ===
def load_band(filename):
    path = os.path.join(base_path, filename + ".tif")
    return np.array(Image.open(path))

def normalize_band(band, max_val=8000):
    return np.where(band > max_val, 1.0, band / max_val)

B02_raw = load_band("T23KMQ_20250604T131239_B02_20m(adaptado)")
B03_raw = load_band("T23KMQ_20250604T131239_B03_20m(adaptado)")
B04_raw = load_band("T23KMQ_20250604T131239_B04_20m(adaptado)")
SCL     = load_band("T23KMQ_20250604T131239_SCL_20m(adaptado)")

B02 = normalize_band(B02_raw)
B03 = normalize_band(B03_raw)
B04 = normalize_band(B04_raw)
mask_full = np.isin(SCL, [4, 5, 6])

# === Classificação por blocos paralelos ===
def processar_bloco(i, bloco_linhas, B02, B03, B04, mask_full, modelo):
    i_fim = min(i + bloco_linhas, B02.shape[0])
    bloco_mask = mask_full[i:i_fim, :]
    if not np.any(bloco_mask):
        return i, None

    bloco_B04 = B04[i:i_fim, :].copy()[bloco_mask]
    bloco_B03 = B03[i:i_fim, :].copy()[bloco_mask]
    bloco_B02 = B02[i:i_fim, :].copy()[bloco_mask]
    X_bloco = np.stack([bloco_B04, bloco_B03, bloco_B02], axis=1)

    y_pred = modelo.predict(X_bloco)
    return i, (bloco_mask, y_pred)

def classificar_imagem(modelo, nome_base, bloco_linhas=50, n_threads=12):
    h, w = SCL.shape
    resultado = np.full((h, w), 255, dtype=np.uint8)
    blocos = list(range(0, h, bloco_linhas))

    print(f"[INFO] Classificando {len(blocos)} blocos com {n_threads} threads...")
    resultados = Parallel(n_jobs=n_threads)(
        delayed(processar_bloco)(i, bloco_linhas, B02, B03, B04, mask_full, modelo)
        for i in tqdm(blocos, desc="Blocos")
    )

    for i, res in resultados:
        if res is None:
            continue
        bloco_mask, y_pred = res
        i_fim = min(i + bloco_linhas, h)
        resultado[i:i_fim, :][bloco_mask] = y_pred

    # === Salvar TIFF
    tiff_path = os.path.join(output_img_path, f"{nome_base}.tif")
    Image.fromarray(resultado).save(tiff_path)

    # === Salvar PNG com legenda
    class_colors = ["green", "orange", "blue", "gray"]
    cmap = ListedColormap(class_colors)
    plt.figure(figsize=(10, 10))
    plt.imshow(resultado, cmap=cmap, vmin=0, vmax=3)
    plt.axis("off")
    patches = [
        Patch(color=class_colors[i], label=label)
        for i, label in enumerate(["Vegetação", "Não Vegetação", "Água", "Não Definida"])
    ]
    plt.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.savefig(os.path.join(output_img_path, f"{nome_base}.png"), bbox_inches='tight')
    plt.close()

# === Execução ===
modelo = joblib.load(os.path.join(modelo_path, "svm_rbf_model.pkl"))
inicio = time.time()
classificar_imagem(modelo, "classificado_svm", bloco_linhas=50, n_threads=12)
fim = time.time()
print(f"[INFO] Classificação concluída em {fim - inicio:.2f} segundos.")

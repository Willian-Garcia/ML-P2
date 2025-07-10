import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Caminhos
base_path = "./imagens"
modelo_path = "./modelo"
output_path = "./imagens_classificadas"
relatorio_path = os.path.join(output_path, "relatorios", "classificao_rf")
os.makedirs(relatorio_path, exist_ok=True)

# Funções auxiliares
def load_band(filename):
    path = os.path.join(base_path, filename + ".tif")
    img_pil = Image.open(path)
    return np.array(img_pil)

def normalize_band(band, max_val=8000):
    return np.where(band > max_val, 1.0, band / max_val)

def gerar_imagem_classificada_com_legenda_varredura(modelo, nome_base, bloco_linhas=100):
    h, w = SCL.shape
    resultado = np.full((h, w), 255, dtype=np.uint8)
    total_blocos = (h + bloco_linhas - 1) // bloco_linhas
    todas_preds = []

    for idx, i in enumerate(range(0, h, bloco_linhas), start=1):
        i_fim = min(i + bloco_linhas, h)
        bloco_mask = mask_full[i:i_fim, :]
        if not np.any(bloco_mask):
            continue

        bloco_B04 = B04[i:i_fim, :][bloco_mask]
        bloco_B03 = B03[i:i_fim, :][bloco_mask]
        bloco_B02 = B02[i:i_fim, :][bloco_mask]
        X_bloco = np.stack([bloco_B04, bloco_B03, bloco_B02], axis=1)

        y_pred_bloco = modelo.predict(X_bloco)
        todas_preds.extend(y_pred_bloco)
        resultado[i + np.where(bloco_mask)[0], np.where(bloco_mask)[1]] = y_pred_bloco

        print(f"[INFO] Bloco {idx}/{total_blocos} classificado.")

    # Salvar .tif
    img_out = Image.fromarray(resultado)
    img_out.save(os.path.join(output_path, f"{nome_base}.tif"))

    # Gerar legenda
    class_colors = ["green", "orange", "blue", "gray"]
    cmap = ListedColormap(class_colors)
    plt.figure(figsize=(10, 10))
    plt.imshow(resultado, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"Classificação - {nome_base}")
    plt.axis("off")
    legend_labels = ["Vegetação", "Não Vegetação", "Água", "Não Definida"]
    patches = [Patch(color=class_colors[i], label=legend_labels[i]) for i in range(4)]
    plt.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.savefig(os.path.join(output_path, f"{nome_base}.png"), bbox_inches='tight')
    plt.close()

    # Histograma
    contagem = Counter(todas_preds)
    classes = ["Vegetação", "Não Vegetação", "Água"]
    valores = [contagem.get(0, 0), contagem.get(1, 0), contagem.get(2, 0)]

    plt.figure(figsize=(6, 4))
    plt.bar(classes, valores, color=["green", "orange", "blue"])
    plt.title("Distribuição das Classes Preditas")
    plt.ylabel("Número de Pixels")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(relatorio_path, "histograma_classes.png"))
    plt.close()

    # Resumo em .txt
    with open(os.path.join(relatorio_path, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("==== RELATÓRIO DE CLASSIFICAÇÃO - RANDOM FOREST ====\n\n")
        f.write(f"Imagem classificada: {nome_base}.tif\n\n")
        f.write("Contagem por classe:\n")
        for nome, valor in zip(classes, valores):
            f.write(f"{nome}: {valor}\n")

    print("[INFO] Relatórios e imagens salvos com sucesso.")

# Carregar bandas
print("[INFO] Carregando bandas...")
B02_raw = load_band("T23KMQ_20250604T131239_B02_20m(adaptado)")
B03_raw = load_band("T23KMQ_20250604T131239_B03_20m(adaptado)")
B04_raw = load_band("T23KMQ_20250604T131239_B04_20m(adaptado)")
SCL     = load_band("T23KMQ_20250604T131239_SCL_20m(adaptado)")

B02 = normalize_band(B02_raw)
B03 = normalize_band(B03_raw)
B04 = normalize_band(B04_raw)
mask_full = np.isin(SCL, [4, 5, 6])

# Carregar modelo
modelo = joblib.load(os.path.join(modelo_path, "random_forest.pkl"))

# Executar classificação
gerar_imagem_classificada_com_legenda_varredura(modelo, "classificado_rf")

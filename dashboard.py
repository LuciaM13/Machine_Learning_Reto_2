import pandas as pd
import numpy as np
import panel as pn
import hvplot.pandas  # for interactive plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Inicializar Panel
pn.extension(sizing_mode="stretch_width", template="fast")

# --- Cargar datos ---
print("Cargando datos...")
df = pd.read_csv("./00_Data/Clean/train.csv")
print("Datos cargados:", df.shape)

# --- PCA ---
def compute_pca(df, n_components=None):
    features = [c for c in df.columns if c != 'Class']
    X_std = StandardScaler().fit_transform(df[features])
    pca = PCA(n_components=n_components or len(features))
    pcs = pca.fit_transform(X_std)
    explained = pca.explained_variance_ratio_
    return pcs, explained

pcs, explained_var = compute_pca(df)

# --- Widgets ---
var_selector = pn.widgets.Select(name='Variable', options=[c for c in df.columns if c != 'Class'])
bins_slider = pn.widgets.IntSlider(name='Bins', start=10, end=100, step=5, value=30)

# --- Gráficos ---
def univariate_plots(var, bins):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[var], bins=bins, kde=True, ax=axs[0])
    axs[0].set_title(f"Histograma - {var}")
    sns.boxplot(x=df[var], ax=axs[1], orient='h')
    axs[1].set_title(f"Boxplot - {var}")
    plt.tight_layout()
    return fig

def bivariate_density(var):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=df[df["Class"] == 0], x=var, fill=True, label="No Fraude", alpha=0.5, ax=ax)
    sns.kdeplot(data=df[df["Class"] == 1], x=var, fill=True, label="Fraude", alpha=0.5, ax=ax)
    ax.set_title(f"Densidad por Clase - {var}")
    ax.legend()
    return fig

def heatmap_spearman_filtered():
    corr_matrix = df.corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    filtered_corr = corr_matrix.where(abs(corr_matrix) > 0.2)
    fig, ax = plt.subplots(figsize=(13, 8), dpi=100)
    sns.heatmap(filtered_corr, mask=mask, cmap='coolwarm', center=0, annot=True, fmt=".1f", linewidths=0.5, ax=ax)
    ax.set_title("Correlación de Spearman (mayores a 0.2)")
    return fig

def heatmap_class_corr():
    corr = df.corr(method='spearman')[['Class']].sort_values(by='Class', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-0.8, vmax=0.8, fmt=".2f", ax=ax)
    ax.set_title("Correlación de Spearman - > 0.5 con 'class'")
    return fig

def pca_plot():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(1, len(explained_var)+1), explained_var)
    ax[0].set_title("Scree Plot")
    ax[0].set_xlabel("Componente")
    ax[0].set_ylabel("Varianza explicada")
    ax[1].plot(np.cumsum(explained_var), marker='o')
    ax[1].set_title("Varianza Acumulada")
    ax[1].set_xlabel("Componentes")
    ax[1].set_ylim(0, 1.05)
    for thr in [0.8, 0.9, 0.95]:
        ax[1].axhline(thr, ls="--", color="gray")
    plt.tight_layout()
    return fig

# --- Paneles dinámicos ---
univar_panel = pn.bind(univariate_plots, var=var_selector, bins=bins_slider)
bivar_panel = pn.bind(bivariate_density, var=var_selector)

# --- Tabs ---
multi_tab = pn.Column(
    pn.pane.Markdown("### Análisis de correlación de Spearman"),
    heatmap_class_corr,
    pn.Spacer(height=20),
    heatmap_spearman_filtered
)

tabs = pn.Tabs(
    ("Univariante", pn.Column(var_selector, bins_slider, univar_panel)),
    ("Bivariante", pn.Column(var_selector, bivar_panel)),
    ("Multivariante", multi_tab),
    ("PCA", pn.Column(pca_plot))
)

# --- Lanzar dashboard ---
tabs.servable()



#Para activarlo ejecutar: y despues en la terminal poner:

#panel serve dashboard.py --show
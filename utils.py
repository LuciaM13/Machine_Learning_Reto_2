import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from scipy.stats import ttest_ind
import os
from typing import Dict, List, Tuple


def plot_density_with_ttest(
    df: pd.DataFrame,
    columns: list[str],
    target: str = "Class",
    figsize: tuple[int, int] = (10, 5),
):
    """
    Para cada columna en `columns` dibuja dos KDEs (por cada clase) y anota el resultado
    de un t-test de Welch (p-valor y nivel de significancia) en el mismo gráfico.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables y la columna target.
    columns : list[str]
        Lista de nombres de columnas numéricas a comparar.
    target : str
        Nombre de la columna objetivo binaria.
    figsize : tuple[int, int]
        Tamaño base (ancho, alto) de cada subplot.

    Ejemplo de uso
    --------------
    plot_density_with_ttest(df, num_columns, target="Class", figsize=(10, 4))
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_ind

    n = len(columns)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize[0], figsize[1] * n))
    # Forzar que axes sea iterable
    if n == 1:
        axes = [axes]

    # Ordenar los valores únicos de la clase para consistencia
    clases = sorted(df[target].unique())

    for ax, col in zip(axes, columns):
        # KDE por cada clase
        for cls in clases:
            sns.kdeplot(
                data=df[df[target] == cls],
                x=col,
                ax=ax,
                label=f"{target} = {cls}",
                fill=True,
                alpha=0.6
            )

        # T-test de Welch
        grp0 = df[df[target] == clases[0]][col].dropna()
        grp1 = df[df[target] == clases[1]][col].dropna()
        _, p_val = ttest_ind(grp0, grp1, equal_var=False)

        # Nivel de significancia
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # Anotar en el gráfico
        ax.text(
            0.95, 0.95,
            f"p = {p_val:.3f} {sig}",
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
        )

        ax.set_title(f"Densidad de {col} por {target}")
        ax.set_xlabel(col)
        ax.set_ylabel("Densidad")
        ax.legend()

    plt.tight_layout()
    plt.show()




def analisis_univariado_numerico(
    df: pd.DataFrame,
    transform_cols: dict[str, str] | None = None,
    bins: int = 40,
):
    """
    Muestra, para cada variable numérica:
      • Histograma + KDE
      • Boxplot
    Si la variable aparece en `transform_cols`, aplica la transformación indicada
    ('log', 'boxcox' o 'yeo') y grafica también la versión transformada.

    Ejemplo:
    --------
    analisis_univariado_numerico(
        df,
        transform_cols={"Amount": "log", "Time": "yeo"}
    )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import boxcox, ttest_ind
    from sklearn.preprocessing import PowerTransformer
    import numpy as np

    transform_cols = transform_cols or {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Preparador para Yeo-Johnson (una sola instancia sirve para todas)
    yeo = PowerTransformer(method="yeo-johnson", standardize=False)

    for col in num_cols:
        original = df[col].dropna().values
        tform = transform_cols.get(col, None)

        # --- aplicar transformación si procede ----------------------------
        if tform == "log":
            transformed = np.log1p(np.clip(original, -0.999999, None))
        elif tform == "boxcox":
            if (original <= 0).any():
                print(f"{col}: Box-Cox omitida porque hay valores ≤ 0")
                transformed = None
            else:
                transformed, _ = boxcox(original)
        elif tform == "yeo":
            transformed = yeo.fit_transform(original.reshape(-1, 1)).ravel()
        else:
            transformed = None

        # --- plot ---------------------------------------------------------
        if transformed is None:          # solo original
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(original, bins=bins, kde=True, ax=ax1, color="C0")
            ax1.set_title(f"Histograma / KDE – {col}")
            sns.boxplot(x=original, ax=ax2, orient="h", color="C0")
            ax2.set_title(f"Boxplot – {col}")
        else:                            # original y transformado
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            sns.histplot(original, bins=bins, kde=True, ax=axes[0], color="C0")
            axes[0].set_title(f"Histograma – {col}")
            sns.histplot(transformed, bins=bins, kde=True, ax=axes[1], color="C1")
            axes[1].set_title(f"Histograma – {tform}({col})")
            # Boxplots enfrentados
            sns.boxplot(
                data=pd.DataFrame({col: original, f"{tform}({col})": transformed}),
                orient="h",
                ax=axes[2],
                palette={"C0", "C1"}
            )
            axes[2].set_title(f"Boxplot – {col} vs {tform}({col})")

        plt.tight_layout()
        plt.show()




def comparativa_cortes_manual(
    df: pd.DataFrame,
    cortes_manual: dict,
    output_dir: str | None = None,
    fmt: str = "png",
):
    """
    Para cada variable indicada en 'cortes_manual' muestra y guarda (si se indica
    'output_dir') una figura con:
      – Histograma + KDE antes del corte
      – Histograma + KDE después del corte
      – Boxplot antes del corte
      – Boxplot después del corte
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    saved_paths = []

    for var, limites in cortes_manual.items():
        df_filt = df.copy()
        if limites.get("sup") is not None:
            df_filt = df_filt[df_filt[var] <= limites["sup"]]
        if limites.get("inf") is not None:
            df_filt = df_filt[df_filt[var] >= limites["inf"]]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(
            f"Comparativa antes y después de aplicar cortes en '{var}'", fontsize=16
        )

        sns.histplot(df[var], kde=True, ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title(f"Histograma Antes – {var}")

        sns.histplot(df_filt[var], kde=True, ax=axes[0, 1], color="green")
        axes[0, 1].set_title(f"Histograma Después – {var}")

        sns.boxplot(x=df[var], ax=axes[1, 0], color="skyblue", orient="h")
        axes[1, 0].set_title(f"Boxplot Antes – {var}")

        rng_inf = limites.get("inf", "-∞")
        rng_sup = limites.get("sup", "∞")
        sns.boxplot(x=df_filt[var], ax=axes[1, 1], color="green", orient="h")
        axes[1, 1].set_title(
            f"Boxplot Después – {var} (Rango [{rng_inf}, {rng_sup}])"
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        if output_dir:
            fname = f"{var}_cortes.{fmt}"
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=150)
            saved_paths.append(path)
            plt.close(fig)

    return saved_paths




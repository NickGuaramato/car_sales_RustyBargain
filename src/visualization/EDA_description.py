#EDA_description

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.config_manager import load_paths

PATHS_DIRS = load_paths()["dirs"]
PATHS_FILES = load_paths()["files"]

"""def load_data() -> dict:
    return {
        "raw": pd.read_csv((PATHS_FILES["raw_data"])),
        "no_duplicates": pd.read_pickle((PATHS_DIRS["metrics"]) / "unduplicated_data.pkl"),
        "cleaned": pd.read_pickle((PATHS_DIRS["metrics"]) / "preprocessed_data.pkl"),
        "final": pd.read_parquet(PATHS_FILES["processed_data"])
        }"""

def validating_cols(df: pd.DataFrame, required_cols: list):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes para EDA: {missing}")

def generate_stats(df: pd.DataFrame, name: str) -> None:
    "guarda estadstica descriptiva"
    stats = df.describe(include='all')
    stats.to_csv(Path(PATHS_DIRS["metrics"]) / f"{name}_stats.csv", index=True)

def plot_categorical_distributions(df: pd.DataFrame, name: str) -> None:
    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type', 'not_repaired', 'brand']
    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='viridis')
            plt.title(f'Distribución de {col} ({name})')
            plt.xticks(rotation=90)
            plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_{col}_distribution.png")
            plt.close()

def plot_numeric_distributions(df: pd.DataFrame, name: str) -> None:
    numeric_cols = ['registration_year', 'power', 'registration_month', 'price']
    for col in numeric_cols:
        if col in df.columns:
            #Boxplot (log scale)
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=np.log1p(df[col]))
            plt.title(f'Distribución de {col} (log) ({name})')
            plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_boxplot_log_{col}.png")
            plt.close()
            
            #Histograma
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f'Histograma de {col} ({name})')
            plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_hist_{col}.png")
            plt.close()

def plot_price_analysis(df: pd.DataFrame, name: str) -> None:
    if 'price' in df.columns:
        #Histograma comparativo (original vs log)
        log_price = np.log1p(df['price'])
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['price'], kde=True, color='blue')
        plt.title('Precios originales')
        
        plt.subplot(1, 2, 2)
        sns.histplot(log_price, kde=True, color='green')
        plt.title('Precios (log)')
        
        plt.suptitle(f'Distribución de precios ({name})')
        plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_price_comparison.png")
        plt.close()

def plot_correlation_matrix(df: pd.DataFrame, name: str) -> None:
    """Matriz de correlación"""
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"Matriz de Correlación ({name})")
        plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_corr_matrix.png")
        plt.close()

def plot_top50_models(df: pd.DataFrame, name: str) -> None:
    if 'model' in df.columns:
        top50 = df['model'].value_counts().head(50).index
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df[df['model'].isin(top50)], x='model', order=top50)
        plt.xticks(rotation=90)
        plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_top50_distribution.png")
        plt.close()

def analyze_outliers_iqr(df: pd.DataFrame, name: str) -> None:
    numeric_cols = ['registration_year', 'power', 'registration_month']
    outliers_info = []
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            count = len(outliers)
            print(f'{name} - {col}: {count} outliers detectados')
            outliers_info.append({'columna': col, 'outliers': count, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR})
    
    #Guardar CSV
    if outliers_info:
        outliers_df = pd.DataFrame(outliers_info)
        outliers_df.to_csv(Path(PATHS_DIRS["metrics"]) / f"{name}_outliers_iqr.csv", index=False)

def analyze_atypical_values(df: pd.DataFrame, name: str) -> None:
    """Valores atípicos específicos (ej: registration_month < 1)"""
    if 'registration_month' in df.columns:
        outliers = df[df['registration_month'] < 1]
        print(f'{name} - registration_month < 1: {len(outliers)} valores')

def analyze_price_by_category(df: pd.DataFrame, name: str) -> None:
    categorical_cols = ['vehicle_type', 'fuel_type', 'gearbox', 'brand']
    for col in categorical_cols:
        if col in df.columns and 'price' in df.columns:
            mean_price = df.groupby(col)['price'].mean().sort_values(ascending=False)
            # Guardar tabla CSV (con Path)
            mean_price.to_csv(Path(PATHS_DIRS["metrics"]) / f"{name}_price_by_{col}.csv")
            # Generar gráfico (con Path)
            plt.figure(figsize=(10, 6))
            mean_price.plot(kind='bar')
            plt.title(f'Precio promedio por {col} ({name})')
            plt.xticks(rotation=90)
            plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_price_by_{col}.png")
            plt.close()

def plot_atypical_distribution(df: pd.DataFrame, name: str) -> None:
    """Distribución atípica específica (como en monolito)"""
    if 'registration_month' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df['registration_month'], bins=12, kde=False)
        plt.title(f'Distribución atípica registration_month ({name})')
        plt.xlabel('Mes')
        plt.ylabel('Frecuencia')
        plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_atypical_month_dist.png")
        plt.close()

def plot_price_boxplots_by_category(df: pd.DataFrame, name: str) -> None:
    """Boxplots de precio por categoría (como en monolito)"""
    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type']
    for col in categorical_cols:
        if col in df.columns and 'price' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=col, y='price', palette='viridis')
            plt.title(f'Distribución del precio por {col} ({name})')
            plt.xlabel(col)
            plt.ylabel('Precio')
            plt.xticks(rotation=90)
            plt.savefig(Path(PATHS_DIRS["plots"]) / f"{name}_price_boxplot_{col}.png")
            plt.close()

def check_columns(df: pd.DataFrame, name: str) -> None:
    #Verifica columnas esperadas vs disponibles
    expected_cols = {
        'target': ['price'],
        'categorical': ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'model', 'not_repaired'],
        'numeric': ['registration_year', 'power', 'mileage', 'registration_month'],
        'derived': ['vehicle_age', 'mileage_per_year']  # Para datasets finales
    }
    
    print(f"\n[{name}] Columnas disponibles:")
    for category, cols in expected_cols.items():
        available = [col for col in cols if col in df.columns]
        if available:
            print(f"  {category}: {len(available)}/{len(cols)}")
            if len(available) < len(cols):
                missing = set(cols) - set(available)
                print(f"    Faltan: {missing}")

def run_full_eda():
    datasets = {
        "raw": PATHS_FILES["raw_data"],
        "no_duplicates": PATHS_DIRS["metrics"] / "unduplicated_data.pkl",
        "cleaned": PATHS_DIRS["metrics"] / "preprocessed_data.pkl",
        "final": PATHS_FILES["processed_data"]
    }

    loaded_data = {}

    for name, path in datasets.items():
        print(f"Cargando {name}...")
        try:
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            elif path.suffix == '.pkl':
                df = pd.read_pickle(path)
            elif path.suffix == '.parquet':
                df = pd.read_parquet(path)
            
            loaded_data[name] = df
            print(f"  ✓ {len(df)} registros cargados")
        except Exception as e:
            print(f"  ✗ Error cargando {name}: {e}")
            continue

    for name, df in loaded_data.items():
        print(f"\nProcesando {name}...")

        check_columns(df, name)

        generate_stats(df, name)
        plot_categorical_distributions(df, name)
        plot_numeric_distributions(df, name)
        plot_price_analysis(df, name)
        plot_correlation_matrix(df, name)
        plot_top50_models(df, name)
        analyze_outliers_iqr(df, name)
        analyze_atypical_values(df, name)
        analyze_price_by_category(df, name)
        plot_atypical_distribution(df, name)
        plot_price_boxplots_by_category(df, name)

        print(f"  ✓ {name} procesado correctamente")

if __name__ == "__main__":
    run_full_eda()
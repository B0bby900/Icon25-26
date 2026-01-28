from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

class Learner:
    def __init__(self, df):
        self.df = df
        warnings.filterwarnings(action="ignore", category=UserWarning)

    def exploratory_analysis(self):
        """
        Genera i grafici citati nella Sezione 1.1 della Documentazione:
        - Heatmap di Correlazione
        - Istogramma Volumi
        """
        print("Generazione grafici EDA (Sez 1.1 Documentazione)...")
        
        # 1. Heatmap Correlazione
        plt.figure(figsize=(10, 8))
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Indice_Volatilita', 'Rendimento_Log']
        # Filtra solo colonne esistenti
        valid_cols = [c for c in cols if c in self.df.columns]
        corr_matrix = self.df[valid_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Heatmap di Correlazione (Analisi Preliminare)')
        
        # 2. Istogramma Volumi
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Volume'], bins=50, kde=True, color='blue')
        plt.title('Distribuzione dei Volumi (Coda Lunga)')
        plt.xlabel('Volume')
        plt.ylabel('Frequenza')

    def plot_elbow_method(self, max_k=10):
        features = ['Rendimento_Giornaliero', 'Volume_Normalizzato', 'Indice_Volatilita']
        available = [f for f in features if f in self.df.columns]
        
        if not available: return

        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[available])
        
        sse = []
        K_range = range(1, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(K_range, sse, marker='o', linestyle='--')
        plt.title('Metodo del Gomito (Elbow Method)')
        plt.xlabel('Numero di Cluster (k)')
        plt.ylabel('Inerzia')
        plt.grid(True)

    def unsupervised_clustering(self, k=4):
        features = ['Rendimento_Giornaliero', 'Volume_Normalizzato', 'Indice_Volatilita']
        available = [f for f in features if f in self.df.columns]
        
        if len(available) < len(features):
            return self.df

        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[available])
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        self.df['Cluster'] = kmeans.fit_predict(X)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df, x='Rendimento_Giornaliero', y='Volume_Normalizzato', 
            hue='Cluster', palette='viridis', s=50
        )
        plt.title(f'K-Means Clustering (k={k}) - Regimi di Mercato')
        plt.grid(True, alpha=0.5)
        
        return self.df

    def supervised_classification(self):
        """Classificazione Trend con Decision Tree."""
        feature_cols = ['Close', 'Volume_Normalizzato', 'SMA_7', 'SMA_30', 'Indice_Volatilita', 'Rendimento_Giornaliero']
        
        X = self.df[feature_cols]
        y = self.df['Target_Profitto']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print("\n=== Report Classificazione (Previsione Trend) ===")
        print(classification_report(y_test, y_pred, target_names=['Giù/Neutro', 'Su']))
        
        # Opzionale: Visualizza l'albero (utile per il report)
        plt.figure(figsize=(12, 8))
        plot_tree(clf, feature_names=feature_cols, class_names=['No Profit', 'Profit'], filled=True, max_depth=2)
        plt.title("Decision Tree (Primi livelli)")

    def supervised_regression(self):
        """Regressione Prezzo Futuro (Ridge) - CORRETTA con Scaling."""
        df_reg = self.df.copy()
        df_reg['Prezzo_Target'] = df_reg['Close'].shift(-1)
        df_reg.dropna(inplace=True)
        
        feature_cols = ['Close', 'SMA_7', 'SMA_30', 'Indice_Volatilita', 'Volume_Normalizzato']
        
        X = df_reg[feature_cols]
        y = df_reg['Prezzo_Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # [CORREZIONE] Pipeline: Standardizzazione -> Ridge
        # Senza StandardScaler, la Ridge non funziona bene su variabili con scale diverse (Vol vs Prezzo)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n=== Report Regressione (Prezzo Futuro) ===")
        print(f"MSE Ridge: {mse:.4f}")
        print(f"R2 Score:  {r2:.4f}")

        plt.figure(figsize=(12, 6))
        x_ax = range(len(y_test))
        plt.plot(x_ax, y_test, label="Reale", color="blue", linewidth=1)
        plt.plot(x_ax, y_pred, label="Predizione Ridge", color="red", linestyle="--", linewidth=1)
        plt.title("Previsione Prezzo Facebook (Ridge Regression)")
        plt.legend()
        plt.grid(True, alpha=0.3)
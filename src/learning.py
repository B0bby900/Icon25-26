from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Learner:
    def __init__(self, df):
        self.df = df

    def unsupervised_clustering(self, k=4):
        """Esegue K-Means per identificare regimi di mercato."""
        features = ['Log_Return', 'Volatility_Idx', 'SMA_7', 'SMA_30']
        
        available_features = [f for f in features if f in self.df.columns]
        
        if not available_features:
            print("Feature per clustering mancanti.")
            return self.df

        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[available_features])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X)
        
        # Visualizzazione (opzionale)
        plt.figure(figsize=(10,6))
        if 'Log_Return' in self.df.columns and 'Volatility_Idx' in self.df.columns:
                sns.scatterplot(data=self.df, x='Log_Return', y='Volatility_Idx', hue='Cluster', palette='viridis')
                plt.title('Clustering dei Regimi di Mercato (K-Means)')
                plt.show()
        
        return self.df

    def supervised_prediction(self):
        """Addestra un Albero di Decisione e mostra un report con etichette finanziarie."""
        features = ['Log_Return', 'Volatility_Idx', 'SMA_7', 'SMA_30', 'Volume']
        target = 'Target_Profit'
        
        valid_features = [f for f in features if f in self.df.columns]
        
        if not valid_features or target not in self.df.columns:
            print("Dati insufficienti per Supervised Learning")
            return None

        X = self.df[valid_features]
        y = self.df[target]
        
        # Split temporale (non shufflato)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        clf = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # --- GENERAZIONE TABELLA ESTETICA ---
        print("\n=== Report Predittivo (Decision Tree) ===")
        
        # 1. Otteniamo il dizionario grezzo
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # 2. Convertiamo in DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        
        # 3. Rinominiamo le RIGHE (Indici) per renderle leggibili
        # '0' = Prezzo scende o stabile, '1' = Prezzo sale
        try:
            report_df.rename(index={
                '0': 'Trend RIBASSISTA (No Profit)',
                '1': 'Trend RIALZISTA (Profit)',
                'accuracy': 'ACCURATEZZA TOTALE',
                'macro avg': 'Media Semplice',
                'weighted avg': 'Media Pesata'
            }, inplace=True)
        except:
            pass # Se le chiavi sono diverse (es. float), ignora e stampa comunque

        # 4. Rinominiamo le COLONNE
        report_df.rename(columns={
            'precision': 'Precisione (Affidabilità)',
            'recall': 'Recall (Sensibilità)',
            'f1-score': 'Punteggio F1',
            'support': 'Giorni Osservati'
        }, inplace=True)
        
        # Formattazione per mostrare 4 decimali
        pd.set_option('display.precision', 4)
        
        print(report_df)
        print("=========================================")
        
        return clf

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Learner:
    def __init__(self, df):
        self.df = df

    def unsupervised_clustering(self, k=4):
        """Esegue K-Means per identificare regimi di mercato."""
        # CORRETTO: Inserite le feature per il clustering
        features = ['Log_Return', 'Volatility_Idx', 'SMA_7', 'SMA_30']
        
        # Controllo se le colonne esistono per evitare errori
        available_features = [f for f in features if f in self.df.columns]
        
        if not available_features:
            print("Feature per clustering mancanti.")
            return self.df

        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[available_features])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X)
        
        # Visualizzazione
        plt.figure(figsize=(10,6))
        # Verifica esistenza colonne per plot
        if 'Log_Return' in self.df.columns and 'Volatility_Idx' in self.df.columns:
            sns.scatterplot(data=self.df, x='Log_Return', y='Volatility_Idx', hue='Cluster', palette='viridis')
            plt.title('Clustering dei Regimi di Mercato (K-Means)')
            plt.show()
        
        return self.df

    def supervised_prediction(self):
        """Addestra un Albero di Decisione per predire il movimento."""
        # CORRETTO: Inserite le feature per la predizione
        features = ['Log_Return', 'Volatility_Idx', 'SMA_7', 'SMA_30', 'Volume']
        target = 'Target_Profit'
        
        # Filtra feature valide
        valid_features = [f for f in features if f in self.df.columns]
        
        if not valid_features or target not in self.df.columns:
            print("Dati insufficienti per Supervised Learning")
            return None

        X = self.df[valid_features]
        y = self.df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        clf = DecisionTreeClassifier(max_depth=5, criterion='entropy')
        clf.fit(X_train, y_train)
        
        acc = clf.score(X_test, y_test)
        print(f"Accuratezza Albero di Decisione: {acc:.4f}")
        return clf

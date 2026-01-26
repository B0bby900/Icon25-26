from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class Learner:
    def __init__(self, df):
        self.df = df

    def unsupervised_clustering(self, k=4):
        """Esegue K-Means per identificare regimi di mercato."""
        features =
        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[features])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X)
        
        # Visualizzazione (Richiesta dal prompt)
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=self.df, x='Log_Return', y='Volatility_Idx', hue='Cluster', palette='viridis')
        plt.title('Clustering dei Regimi di Mercato (K-Means)')
        plt.show()
        
        return self.df

    def supervised_prediction(self):
        """Addestra un Albero di Decisione per predire il movimento."""
        X = self.df]
        y = self.df
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        clf = DecisionTreeClassifier(max_depth=5, criterion='entropy')
        clf.fit(X_train, y_train)
        
        acc = clf.score(X_test, y_test)
        print(f"Accuratezza Albero di Decisione: {acc:.4f}")
        return clf
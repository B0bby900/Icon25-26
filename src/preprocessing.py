
import pandas as pd
import numpy as np

class DataEngine:
    def __init__(self, config):
        self.ohlc_path = config['dataset']['file_ohlc']
        self.metrics_path = config['dataset']['file_metrics']
        self.df = None

    def load_data(self):
        """Carica e unisce i dataset di Facebook."""
        # Caricamento dati OHLC
        df1 = pd.read_csv(self.ohlc_path)
        df1 = pd.to_datetime(df1)
        
        # Caricamento dati Metriche (con Profitto)
        df2 = pd.read_csv(self.metrics_path)
        df2 = pd.to_datetime(df2)
        
        # Merge dei dataset
        self.df = pd.merge(df1, df2], on='Date', how='left')
        self.df.sort_values('Date', inplace=True)
        
        # Gestione valori mancanti
        self.df.fillna(method='ffill', inplace=True)
        
        # Feature Engineering
        self._add_technical_features()
        
        print(f"Dataset caricato: {len(self.df)} righe.")
        return self.df

    def _add_technical_features(self):
        """Aggiunge indicatori tecnici per l'analisi."""
        # Medie Mobili (SMA)
        self.df = self.df['Close'].rolling(window=7).mean()
        self.df = self.df['Close'].rolling(window=30).mean()
        
        # Volatilità (High - Low) / Open
        self.df['Volatility_Idx'] = (self.df['High'] - self.df['Low']) / self.df['Open']
        
        # Rendimento Logaritmico
        self.df = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Target per Classificazione (1 se Profit > 0)
        self.df = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)
        
        self.df.dropna(inplace=True)

    def get_discrete_data(self):
        """Discretizza le variabili continue per la Rete Bayesiana Discreta."""
        disc_df = self.df.copy()
        
        # Discretizzazione Volume (Terzili)
        disc_df['Volume_Cat'] = pd.qcut(disc_df['Volume'], 3, labels=['Low', 'Medium', 'High'])
        
        # Discretizzazione Volatilità
        disc_df['Vol_Cat'] = pd.qcut(disc_df['Volatility_Idx'], 3, labels=)
        
        # Discretizzazione Profitto
        disc_df['Profit_Cat'] = disc_df['Profit'].apply(lambda x: 'Gain' if x > 0 else 'Loss')
        
        return disc_df

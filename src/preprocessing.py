import pandas as pd
import numpy as np
import os

class DataEngine:
    def __init__(self, config):
        self.ohlc_path = config['dataset']['file_ohlc']
        self.metrics_path = config['dataset']['file_metrics']
        self.df = None

    def load_data(self):
        """Carica e unisce i dataset di Facebook."""
        if not os.path.exists(self.ohlc_path) or not os.path.exists(self.metrics_path):
            print(f"Errore: File dati non trovati ({self.ohlc_path} o {self.metrics_path})")
            return pd.DataFrame()

        try:
            # Caricamento dati OHLC (Prezzi Float)
            df1 = pd.read_csv(self.ohlc_path)
            df1['Date'] = pd.to_datetime(df1['Date']) 
            
            # Caricamento dati Metriche (Profitto, ecc.)
            df2 = pd.read_csv(self.metrics_path)
            df2['Date'] = pd.to_datetime(df2['Date']) 
            
            # Merge: Inner join per mantenere solo le date presenti in entrambi
            # Usiamo suffixes per distinguere colonne omonime (es. Open float vs Open int)
            self.df = pd.merge(df1, df2, on='Date', how='inner', suffixes=('', '_metric'))
            self.df.sort_values('Date', inplace=True)
            self.df.ffill(inplace=True)        
            
            # Feature Engineering
            self._add_technical_features()
            
            self.df.dropna(inplace=True)

            print(f"Dataset caricato e processato: {len(self.df)} righe.")
            return self.df
        except Exception as e:
            print(f"Errore nel caricamento dati: {e}")
            return pd.DataFrame()

    def _add_technical_features(self):
        """Calcola indicatori tecnici."""
        # 1. Medie Mobili
        self.df['SMA_7'] = self.df['Close'].rolling(window=7).mean()
        self.df['SMA_30'] = self.df['Close'].rolling(window=30).mean()
        
        # 2. Indice di Volatilità (High - Low) / Open
        self.df['Indice_Volatilita'] = (self.df['High'] - self.df['Low']) / self.df['Open']
        
        # 3. Rendimento Giornaliero %
        self.df['Rendimento_Giornaliero'] = self.df['Close'].pct_change()
        
        # 4. Volume Normalizzato
        vol_sma_30 = self.df['Volume'].rolling(window=30).mean()
        self.df['Volume_Normalizzato'] = self.df['Volume'] / vol_sma_30
        
        # 5. Rendimento Logaritmico (per stabilità statistica)
        self.df['Rendimento_Log'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Target Classificazione: 1 se Close domani > Close oggi (Trend Rialzista Futuro)
        self.df['Target_Profitto'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)

    def get_discrete_data(self):
        """Discretizza le variabili per la Rete Bayesiana."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        disc_df = self.df.copy()
        
        # Discretizzazione con qcut (quantili)
        disc_df['Cat_Volume'] = pd.qcut(disc_df['Volume'], 3, labels=['Basso', 'Medio', 'Alto'])
        disc_df['Cat_Volatilita'] = pd.qcut(disc_df['Indice_Volatilita'], 2, labels=['Calma', 'Agitata'])
        disc_df['Cat_Profitto'] = pd.qcut(disc_df['Rendimento_Log'], 3, labels=['Perdita', 'Neutro', 'Guadagno'])
        
        # Logica deterministica per il Trend
        disc_df['Cat_Trend'] = np.where(disc_df['SMA_7'] > disc_df['SMA_30'], 'Rialzista', 'Ribassista')
        
        return disc_df
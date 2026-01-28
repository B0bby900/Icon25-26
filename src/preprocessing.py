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
        df1['Date'] = pd.to_datetime(df1['Date']) 
        
        # Caricamento dati Metriche (con Profitto)
        df2 = pd.read_csv(self.metrics_path)
        df2['Date'] = pd.to_datetime(df2['Date']) 
        
        # Merge dei dataset
        self.df = pd.merge(df1, df2, on='Date', how='inner', suffixes=('', '_metric'))
        self.df.sort_values('Date', inplace=True)
        
        # Gestione valori mancanti
        self.df.ffill(inplace=True)        
        
        # Feature Engineering
        self._add_technical_features()
        
        print(f"Dataset caricato: {len(self.df)} righe.")
        return self.df

    def _add_technical_features(self):
        """Aggiunge indicatori tecnici per l'analisi."""
        
        # Medie Mobili (SMA)
        self.df['SMA_7'] = self.df['Close'].rolling(window=7).mean()
        self.df['SMA_30'] = self.df['Close'].rolling(window=30).mean()
        
        # Volatilità (High - Low) / Open
        self.df['Volatility_Idx'] = (self.df['High'] - self.df['Low']) / self.df['Open']
        
        # Rendimento Logaritmico
        self.df['Log_Return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Target per Classificazione (1 se il prezzo di domani è maggiore di oggi)
        self.df['Target_Profit'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)
        
        # Rimuoviamo le righe con NaN generati da rolling e shift
        self.df.dropna(inplace=True)

    def get_discrete_data(self):
        """Discretizza le variabili continue per la Rete Bayesiana Discreta."""
        disc_df = self.df.copy()
        
        # 1. Volume: 3 stati -> {Basso, Medio, Alto}
        disc_df['Volume_Cat'] = pd.qcut(disc_df['Volume'], 3, labels=['Basso', 'Medio', 'Alto'])
        
        # 2. Volatilità: 2 stati -> {Calma, Agitata}
        disc_df['Vol_Cat'] = pd.qcut(disc_df['Volatility_Idx'], 2, labels=['Calma', 'Agitata'])
        
        # 3. Profitto: 3 stati -> {Perdita, Neutro, Guadagno}
        disc_df['Profit_Cat'] = pd.qcut(disc_df['Log_Return'], 3, labels=['Perdita', 'Neutro', 'Guadagno'])
        
        # 4. Trend: Variabile derivata -> {Rialzista, Ribassista}
        disc_df['Trend_Cat'] = np.where(disc_df['SMA_7'] > disc_df['SMA_30'], 'Rialzista', 'Ribassista')
        
        return disc_df

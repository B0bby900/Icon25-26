from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import norm # Necessario per il grafico gaussiano

class BayesEngine:
    def __init__(self, full_df, discrete_df):
        """
        :param full_df: DataFrame originale con valori continui (per Sez. 4.2)
        :param discrete_df: DataFrame discretizzato (per Sez. 4.1)
        """
        self.df_continuous = full_df
        self.df_discrete = discrete_df
        self.model_discrete = None
        
        # Parametri per il modello Gaussiano Lineare (Sez 4.2)
        self.gaussian_params = {}

    # --- SEZIONE 4.1: RETE DISCRETA (Apprendimento Struttura e Parametri) ---
    def build_discrete_network(self):
        logging.getLogger("pgmpy").setLevel(logging.ERROR)

        # Definizione Struttura: Volume -> Volatilità -> Profitto <- Trend
        self.model_discrete = DiscreteBayesianNetwork([
            ('Cat_Volume', 'Cat_Volatilita'),
            ('Cat_Volatilita', 'Cat_Profitto'),
            ('Cat_Trend', 'Cat_Profitto')
        ])
        
        # Conversione tipi per pgmpy (Object/String)
        train_data = pd.DataFrame()
        cols = ['Cat_Volume', 'Cat_Volatilita', 'Cat_Profitto', 'Cat_Trend']
        try:
            for col in cols:
                train_data[col] = self.df_discrete[col].astype(str).astype(object)

            self.model_discrete.fit(train_data, estimator=MaximumLikelihoodEstimator)
            print("Rete Bayesiana Discreta (Sez 4.1) addestrata.")
        except Exception as e:
            print(f"Errore training Bayes Discreto: {e}")

    def inference_discrete(self, volume_state='Alto', vol_state='Agitata', trend_state='Rialzista'):
        """Inferenza Esatta con visualizzazione grafica."""
        if self.model_discrete is None: return

        infer = VariableElimination(self.model_discrete)
        try:
            print(f"\n[Inferenza ESATTA] Probabilità Profitto dato [Vol={volume_state}, Volat={vol_state}, Trend={trend_state}]:")
            q = infer.query(
                variables=['Cat_Profitto'], 
                evidence={'Cat_Volume': volume_state, 'Cat_Volatilita': vol_state, 'Cat_Trend': trend_state}
            )
            print(q)
            
            # Visualizzazione Istogramma Probabilità
            # Recuperiamo nomi stati e valori
            state_names = q.state_names['Cat_Profitto']
            values = q.values
            
            plt.figure(figsize=(8, 5))
            colors = ['red' if 'Perdita' in s else 'green' if 'Guadagno' in s else 'gray' for s in state_names]
            plt.bar(state_names, values, color=colors, alpha=0.7)
            plt.title(f'Probabilità Profitto (Evidence: {vol_state}, {trend_state})')
            plt.ylabel('Probabilità')
            plt.ylim(0, 1)
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        except Exception as e:
            print(f"Errore inferenza discreta: {e}")

    # --- SEZIONE 4.2: RETE CONTINUA (IBRIDA/GAUSSIANA) ---
    def build_continuous_network(self):
        """
        Implementa il modello Linear Gaussian descritto nella Sezione 4.2.
        Modelliamo: Close ~ N(beta0 + beta1*Open + beta2*Volume, sigma^2)
        """
        try:
            # Selezione dati
            X = self.df_continuous[['Open', 'Volume']]
            y = self.df_continuous['Close']

            # Stima dei parametri (Maximum Likelihood per Gaussiane Lineari equivale a OLS)
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Calcolo della varianza residua (sigma^2)
            y_pred = reg.predict(X)
            mse = mean_squared_error(y, y_pred) # Stima della varianza del rumore
            
            # Salvataggio parametri nella Knowledge Base dell'agente
            self.gaussian_params = {
                'intercept': reg.intercept_,
                'coef_open': reg.coef_[0],
                'coef_vol': reg.coef_[1],
                'sigma': np.sqrt(mse) # Deviazione standard
            }
            
            print(f"Rete Bayesiana Continua (Sez 4.2) configurata.")
            print(f"Parametri: Intercetta={reg.intercept_:.2f}, Coef_Open={reg.coef_[0]:.4f}, Sigma={np.sqrt(mse):.2f}")
            
        except Exception as e:
            print(f"Errore training Bayes Continuo: {e}")

    def inference_continuous(self, open_val, volume_val):
        """
        Esegue inferenza restituendo una distribuzione gaussiana sul prezzo futuro e il relativo grafico.
        """
        if not self.gaussian_params:
            print("Modello continuo non addestrato.")
            return

        # Calcolo della media condizionata: E[Close | Open, Volume]
        mu = (self.gaussian_params['intercept'] + 
              self.gaussian_params['coef_open'] * open_val + 
              self.gaussian_params['coef_vol'] * volume_val)
        
        sigma = self.gaussian_params['sigma']
        
        print(f"\n[Inferenza Continua] Previsione Prezzo dato Open={open_val}, Vol={volume_val}:")
        print(f"Distribuzione Gaussiana N(μ={mu:.2f}, σ={sigma:.2f})")
        print(f"Intervallo di Confidenza 95%: [{mu - 1.96*sigma:.2f}, {mu + 1.96*sigma:.2f}]")
        
        # Visualizzazione Grafica (Curva a Campana)
        plt.figure(figsize=(10, 6))
        
        # Generiamo punti x attorno alla media (es. +/- 4 deviazioni standard)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = norm.pdf(x, mu, sigma)
        
        plt.plot(x, y, label=r'PDF Gaussiana\n$\mu={mu:.2f}, \sigma={sigma:.2f}$', color='darkblue')
        
        # Evidenziamo l'intervallo di confidenza 95%
        x_fill = np.linspace(mu - 1.96*sigma, mu + 1.96*sigma, 100)
        y_fill = norm.pdf(x_fill, mu, sigma)
        plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label='95% Confidence Interval')
        
        plt.axvline(mu, color='black', linestyle='--', alpha=0.5, label='Media (Prezzo Atteso)')
        plt.title('Previsione Probabilistica Prezzo (Bayesiano Continuo)')
        plt.xlabel('Prezzo Previsto ($)')
        plt.ylabel('Densità di Probabilità')
        plt.legend()
        plt.grid(True, alpha=0.2)
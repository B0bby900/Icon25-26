from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
import logging

class BayesEngine:
    def __init__(self, discrete_df):
        self.df = discrete_df
        self.model = None

    def build_discrete_network(self):
        """Costruisce e addestra una Rete Bayesiana Discreta."""
        
        # ZITTIAMO I LOG AUTOMATICI DI PGMPY
        logging.getLogger("pgmpy").setLevel(logging.ERROR)

        # Definizione Struttura (Volume -> Volatilità -> Profitto)
        self.model = DiscreteBayesianNetwork([
            ('Volume_Cat', 'Vol_Cat'),
            ('Vol_Cat', 'Profit_Cat')
        ])
        
        # Preparazione dati
        training_data = pd.DataFrame()
        
        try:
            training_data['Volume_Cat'] = self.df['Volume_Cat'].astype(str).astype(object)
            training_data['Vol_Cat'] = self.df['Vol_Cat'].astype(str).astype(object)
            training_data['Profit_Cat'] = self.df['Profit_Cat'].astype(str).astype(object)
        except KeyError as e:
            print(f"ERRORE CRITICO: Manca la colonna {e} nel dataset!")
            return

        # Apprendimento Parametri (MLE)
        self.model.fit(training_data, estimator=MaximumLikelihoodEstimator)
        
        assert self.model.check_model()

    def inference(self, volume_state='High', vol_state='High_Vol'):
        """Esegue inferenza probabilistica."""
        if self.model is None:
            return

        infer = VariableElimination(self.model)
        
        try:
            # Query: Probabilità di Profitto data evidenza
            q = infer.query(variables=['Profit_Cat'], 
                            evidence={'Volume_Cat': volume_state, 'Vol_Cat': vol_state})
            print(f"\nInferenza [Volume={volume_state}, Volatilità={vol_state}]:")
            print(q)
        except Exception as e:
            print(f"Errore durante l'inferenza: {e}")

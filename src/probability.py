from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

class BayesEngine:
    def __init__(self, discrete_df):
        self.df = discrete_df
        self.model = None

    def build_discrete_network(self):
        """Costruisce e addestra una Rete Bayesiana Discreta."""
        # Definizione Struttura (Volume -> Volatilità -> Profitto)
        self.model = BayesianNetwork([
            ('Volume_Cat', 'Vol_Cat'),
            ('Vol_Cat', 'Profit_Cat')
        ])
        
        # Apprendimento Parametri (MLE)
        self.model.fit(self.df, estimator=MaximumLikelihoodEstimator)
        assert self.model.check_model()
        print("Rete Bayesiana addestrata con successo.")

    def inference(self, volume_state='High', vol_state='Volatile'):
        """Esegue inferenza probabilistica."""
        infer = VariableElimination(self.model)
        # Query: Probabilità di Profitto data evidenza
        q = infer.query(variables=['Profit_Cat'], 
                        evidence={'Volume_Cat': volume_state, 'Vol_Cat': vol_state})
        print(f"Inferenza [Volume={volume_state}, Volatilità={vol_state}]:")
        print(q)

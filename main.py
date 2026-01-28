import xml.etree.ElementTree as ET
from src.preprocessing import DataEngine
from src.logic import PrologManager
from src.learning import Learner
from src.probability import BayesEngine

def parse_config(xml_file='config.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    config = {
        'dataset': {
            'file_ohlc': root.find('./dataset/file_ohlc').text,
            'file_metrics': root.find('./dataset/file_metrics').text
        },
        'output': {
            'facts_path': root.find('./output/facts_path').text
        }
    }
    return config

def main():
    print("=== Avvio Agente Finanziario Facebook ===")
    config = parse_config()
    
    # 1. Caricamento e Preprocessing
    engine = DataEngine(config)
    df = engine.load_data()
    
    # 2. Logica e Rappresentazione della Conoscenza
    prolog_mgr = PrologManager(df, config['output']['facts_path'])
    prolog_mgr.generate_facts()
    # patterns = prolog_mgr.query_patterns() # Eseguire se SWI-Prolog è configurato
    
    # 3. Machine Learning (Supervisionato & Non Supervisionato)
    learner = Learner(df)
    df_clustered = learner.unsupervised_clustering() 
    learner.supervised_prediction()
    
    # 4. Ragionamento Probabilistico (Bayesian Network)
    disc_df = engine.get_discrete_data()
    bn_engine = BayesEngine(disc_df)
    bn_engine.build_discrete_network()
    
    # Eseguiamo un'inferenza con i NOMI IN ITALIANO
    # Scenario: Volume Alto, Volatilità Agitata, Trend Rialzista
    bn_engine.inference(volume_state='Alto', vol_state='Agitata', trend_state='Rialzista')
    
if __name__ == "__main__":
    main()

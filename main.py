import xml.etree.ElementTree as ET
import sys
import os
import matplotlib.pyplot as plt 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.preprocessing import DataEngine
    from src.logic import PrologManager
    from src.learning import Learner
    from src.probability import BayesEngine
except ImportError:
    from preprocessing import DataEngine
    from logic import PrologManager
    from learning import Learner
    from probability import BayesEngine

def parse_config(xml_file='config.xml'):
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"File '{xml_file}' non trovato.")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return {
        'dataset': {
            'file_ohlc': root.find('./dataset/file_ohlc').text,
            'file_metrics': root.find('./dataset/file_metrics').text
        },
        'parameters': {
            'kmeans_clusters': int(root.find('./parameters/kmeans_clusters').text),
            'test_size': float(root.find('./parameters/test_size').text)
        },
        'output': {
            'facts_path': root.find('./output/facts_path').text
        }
    }

def main():
    print("=== Avvio Agente Finanziario Facebook ===")
    
    try:
        config = parse_config()
        print("Configurazione caricata.")
    except Exception as e:
        print(f"Errore config: {e}")
        return

    # [Fase 1] Ingestione
    print("\n[Fase 1] Ingestione Dati")
    engine = DataEngine(config)
    df = engine.load_data()
    
    if df.empty: return

    # [Fase 2] Logica Simbolica
    print("\n[Fase 2] Generazione Conoscenza Simbolica")
    prolog_mgr = PrologManager(df, config['output']['facts_path'])
    prolog_mgr.generate_facts()
    prolog_mgr.query_patterns()
    
    # [Fase 3] Machine Learning
    print("\n[Fase 3] Apprendimento Automatico")
    learner = Learner(df)
    
    # Visualizzazioni EDA (Sez 1.1 Doc)
    learner.exploratory_analysis()
    
    learner.plot_elbow_method()
    learner.unsupervised_clustering(k=config['parameters']['kmeans_clusters'])
    learner.supervised_classification()
    learner.supervised_regression()
    
    # [Fase 4] Modelli Probabilistici (Sezione 4 Doc)
    print("\n[Fase 4] Modelli Probabilistici")
    
    # Otteniamo sia i dati discreti (per Sez 4.1) che quelli continui (per Sez 4.2)
    disc_df = engine.get_discrete_data()
    
    # Inizializza BayesEngine con ENTRAMBI i dataset
    bayes = BayesEngine(full_df=df, discrete_df=disc_df)
    
    # --- 4.1 Rete Discreta ---
    bayes.build_discrete_network()
    bayes.inference_discrete(volume_state='Alto', vol_state='Agitata', trend_state='Rialzista')
    
    # --- 4.2 Rete Continua (Linear Gaussian) ---
    bayes.build_continuous_network()
    
    # Inferenza Continua: Prendiamo l'ultimo giorno disponibile come input per test
    last_day = df.iloc[-1]
    bayes.inference_continuous(open_val=last_day['Open'], volume_val=last_day['Volume'])

    print("\n=== Esecuzione Completata ===")
    print("Chiudi le finestre dei grafici per terminare.")
    plt.show()

if __name__ == "__main__":
    main()
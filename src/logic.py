from pyswip import Prolog
import os
import tempfile
import pandas as pd

class PrologManager:
    def __init__(self, df, facts_file):
        self.df = df
        self.facts_file = facts_file
        self.prolog = Prolog()

    def generate_facts(self):
        """Genera facts.pl con dati OHLC e relazioni temporali."""
        try:
            os.makedirs(os.path.dirname(self.facts_file), exist_ok=True)
            
            with open(self.facts_file, 'w', encoding='utf-8') as f:
                f.write("% --- FATTI GENERATI DA PYTHON ---\n")
                f.write(":- dynamic stock_day/7.\n")
                f.write(":- dynamic next_day/2.\n\n")
                
                dates = []
                for _, row in self.df.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d')
                    dates.append(date_str)
                    
                    profit = row['Profit'] if 'Profit' in row else 0.0
                    
                    # stock_day(Data, O, H, L, C, Vol, Profit)
                    fact = f"stock_day('{date_str}', {row['Open']:.2f}, {row['High']:.2f}, {row['Low']:.2f}, {row['Close']:.2f}, {int(row['Volume'])}, {profit:.2f}).\n"
                    f.write(fact)
                
                f.write("\n% --- RELAZIONI TEMPORALI ---\n")
                for i in range(len(dates) - 1):
                    f.write(f"next_day('{dates[i]}', '{dates[i+1]}').\n")
            
            print(f"Knowledge Base aggiornata: {self.facts_file}")
            
        except Exception as e:
            print(f"Errore generazione fatti: {e}")

    def query_patterns(self):
        """Esegue le query Prolog."""
        # [CORREZIONE] Percorso esplicito alle regole statiche
        # Assume che rules.pl sia nella cartella knowledge_base come da struttura progetto
        base_dir = os.path.dirname(self.facts_file)
        rules_file = os.path.join(base_dir, 'rules.pl')
        
        if not os.path.exists(rules_file):
            print(f"ATTENZIONE: File regole '{rules_file}' mancante. Assicurati che rules.pl sia in knowledge_base/")
            return

        try:
            # Creiamo un file temporaneo che unisce Fatti + Regole
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8') as tmp:
                temp_path = tmp.name.replace('\\', '/')
                
                with open(self.facts_file, 'r', encoding='utf-8') as f: tmp.write(f.read() + "\n")
                with open(rules_file, 'r', encoding='utf-8') as f: tmp.write(f.read())
            
            # Reset e Consult
            self.prolog.retractall("stock_day(_,_,_,_,_,_,_)")
            self.prolog.retractall("next_day(_,_)")
            self.prolog.consult(temp_path)
            
            print("\n--- Risultati Ragionamento Logico (Prolog) ---")
            
            # Query: Bullish Engulfing
            results = list(self.prolog.query("engulfing_bullish(Date)"))
            unique_dates = sorted(list(set([res['Date'] for res in results])))
            
            if unique_dates:
                print(f"Pattern 'Bullish Engulfing' trovato in {len(unique_dates)} casi.")
                print("Primi 3 casi:", unique_dates[:3])
            else:
                print("Nessun 'Bullish Engulfing' trovato.")

            # Query: Volatilità Estrema (usa il predicato high_volatility definito in rules.pl)
            try:
                res_vol = list(self.prolog.query("high_volatility(Date)"))
                print(f"Giorni di 'high_volatility': {len(res_vol)}")
            except:
                print("Predicato high_volatility non trovato o errore query.")

            # Cleanup
            try: os.remove(temp_path)
            except: pass

        except Exception as e:
            print(f"Errore Prolog: {e}")
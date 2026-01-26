from pyswip import Prolog

class PrologManager:
    def __init__(self, df, facts_file):
        self.df = df
        self.facts_file = facts_file
        self.prolog = Prolog()

    def generate_facts(self):
        """Converte il DataFrame in fatti Prolog e li salva su file."""
        with open(self.facts_file, 'w') as f:
            f.write(":- dynamic stock_day/7.\n")
            f.write("% stock_day(Date, Open, High, Low, Close, Volume, Profit)\n")
            
            for _, row in self.df.iterrows():
                date_str = row.strftime('%Y-%m-%d')
                # Normalizziamo i float per evitare problemi di precisione
                fact = f"stock_day('{date_str}', {row['Open']:.2f}, {row['High']:.2f}, {row['Low']:.2f}, {row['Close']:.2f}, {int(row['Volume'])}, {row['Profit']:.2f}).\n"
                f.write(fact)
        
        print("Base di Conoscenza generata.")

    def query_patterns(self):
        """Esegue query sulla KB per trovare pattern."""
        self.prolog.consult("knowledge_base/rules.pl") # Carica le regole
        
        # Esempio: Trova giorni di Engulfing Bullish
        engulfing_days = list(self.prolog.query("bullish_engulfing(D1, D2)"))
        return engulfing_days


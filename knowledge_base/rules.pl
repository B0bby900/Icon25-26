% --- REGOLE LOGICHE (Coerenti con Documentazione Sez 2.1) ---

% "Un giorno è 'bullish' se la chiusura è superiore all'apertura"
is_bullish(Date) :-
    stock_day(Date, Open, _, _, Close, _, _),
    Close > Open.

% "Un giorno è 'bearish' se la chiusura è inferiore all'apertura"
is_bearish(Date) :-
    stock_day(Date, Open, _, _, Close, _, _),
    Close < Open.

% Volatilità Significativa (Sez 2.1)
% Si verifica quando lo spread High-Low supera una soglia (es. 5% dell'Open)
high_volatility(Date) :-
    stock_day(Date, Open, High, Low, _, _, _),
    Diff is High - Low,
    Soglia is Open * 0.05,
    Diff > Soglia.

% PATTERN: BULLISH ENGULFING (Sez 2.2)
% Definito come una congiunzione di vincoli geometrici su due giorni (D1=Ieri, D2=Oggi)
engulfing_bullish(DateOggi) :-
    % Recupera dati Oggi (D2)
    stock_day(DateOggi, OpenOggi, _, _, CloseOggi, _, _),
    is_bullish(DateOggi), % Oggi verde
    
    % Recupera il Giorno Precedente esatto (D1)
    next_day(DateIeri, DateOggi),
    stock_day(DateIeri, OpenIeri, _, _, CloseIeri, _, _),
    
    is_bearish(DateIeri), % Ieri rossa
    
    % Condizione di Engulfing (Geometrica)
    % [CORREZIONE] Usiamo =< e >= per includere i casi senza gap estremi
    OpenOggi =< CloseIeri,  % Apre sotto o uguale alla chiusura di ieri
    CloseOggi >= OpenIeri.  % Chiude sopra o uguale all'apertura di ieri

% Carica i fatti generati
:- consult('facts.pl').

% Regola base: Trend Rialzista
is_green(Date) :-
    stock_day(Date, Open, _, _, Close, _, _),
    Close > Open.

is_red(Date) :-
    stock_day(Date, Open, _, _, Close, _, _),
    Close < Open.

% Regola Pattern: Bullish Engulfing
% Richiede due giorni consecutivi D1 (Ieri) e D2 (Oggi)
% Nota: In un sistema reale, servirebbe una relazione next_day(D1, D2). 
% Qui assumiamo che la logica di adiacenza sia gestita passando le date o definendo una lista temporale.

bullish_engulfing(D1, D2) :-
    stock_day(D1, O1, _, _, C1, _, _),
    stock_day(D2, O2, _, _, C2, _, _),
    C1 < O1,           % Ieri: Candela Rossa
    C2 > O2,           % Oggi: Candela Verde
    O2 < C1,           % Apertura Oggi < Chiusura Ieri (Gap Down)
    C2 > O1.           % Chiusura Oggi > Apertura Ieri (Inglobamento)

% Regola Complessa: Segnale di Acquisto Forte
% Engulfing + Volume Alto
strong_buy(D1, D2) :-
    bullish_engulfing(D1, D2),
    stock_day(D2, _, _, _, _, Vol, _),
    Vol > 50000000.    % Soglia di volume (es. 50M)

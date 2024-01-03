Mi serve una funzione che mi restituisca la lista di mosse possibili.
Per farlo devo modificare a catena, all'interno di game.py, le funzioni play, move, take e slide.
Devo aggiungere come parametro un booleano che è false di default (quando devo provare ad eseguire una mossa) e true quando devo calcolare le mosse possibili.
In questo modo, dato un game, date tutte le mosse, posso escludere quelle impossibili.
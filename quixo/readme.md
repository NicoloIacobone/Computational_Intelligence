<!-- Mi serve una funzione che mi restituisca la lista di mosse possibili.
Per farlo devo modificare a catena, all'interno di game.py, le funzioni play, move, take e slide.
Devo aggiungere come parametro un booleano che è false di default (quando devo provare ad eseguire una mossa) e true quando devo calcolare le mosse possibili.
In questo modo, dato un game, date tutte le mosse, posso escludere quelle impossibili. -->

<!-- Oppure potrei aggiungere come attributo della classe Game possible_moves_player_0 e possible_moves_player_1, che sono liste di mosse possibili per i due giocatori. -->

TODO:
- [X] Implementare la funzione `possible_moves`
- [X] Implementare ReinforcementPlayer come in lab 4

Ispirato alla barra di "probabilità di vittoria" di DeepMind che gioca a Starcraft, cercare di avvicinarsi il più possibile ad uno stato di vittoria che viene calcolato facendo la media delle vittorie e delle sconfitte ottenute in quello stato (potrebbe essere utile l'attributo `hit_state`)

Prima versione: perde al 71%, troppo pesante. Circa 2-5 secondi ad iterazione.
Bisogna creare una versione molto più veloce ed ottimizzata.
La policy pesa più di 500MB, troppo.

Seconda versione: migliorato il tempo di esecuzione di circa 10 volte, ora ho una media di 5 iterazioni al secondo. Il miglioramento è stato ottenuto cambiando il modo in cui viene hashata la board, prima hashable_state = tuple(map(tuple, self._board)), ora hashable_state = tuple(self._board.flatten()). Forse un piccolo miglioramento è dato anche dal fatto che nel doppio if che calcola tutte le mosse possibili skippa automaticamente i punti che non sono sul bordo della tavola, quindi prima la funzione .make_one_move() veniva chiamata 25 volte, ora 16, per 5000 giochi in cui vengono fatte decine di mosse a gioco, forse qualche miglioramento è dato anche da questa funzione che è un pò più efficiente.
5000 round di training in 19:16 minuti
create_policy() in 3:13 minuti
la policy pesa comunque mezzo giga

probabilmente c'è un problema con l'hashing della board, analizzando gli hash con i relativi pesi noto che la maggior parte degli stati hanno peso 0.1, molti hanno un valore di 0.19, nessuno ha un valore negativo, pochissimi hanno un valore decimale con più di una cifra dopo la virgola.

Potrei provare a trasformare game._board in una stringa e hashare quella.
Test con questa riga per hashare la board:
hashable_state = np.array2string(new_state._board.flatten())
Inspiegabilmente il numero di iterazioni al secondo è salito a 17.5 di media.
Risultati attesi con 5000 round di training = stessi che in precedenza ma con una policy che in teoria pesa meno, forse ho ottenuto un risultato ragionevole per fare 50k round di training.
policy creata in 0.2 secondi, pesa poco più di 100MB

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

test con 10k round (9:30min):
Win rate: 0.47%
Lose rate: 0.53%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

test con 50k round (47:36min) e exp_rate 0.3:
Win rate: 0.62%
Lose rate: 0.38%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

test con 50k round (47:04min) e exp_rate 0.5 decrescente (-0.1) ogni 10k round:
Win rate: 0.8%
Lose rate: 0.2%

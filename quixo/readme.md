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

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

test con 200k round, exp_rate 0.3, due reinforcement players che giocano uno contro l'altro:
policies: 2.39GB
test vs random player: 
Win rate: 0.3%
Lose rate: 0.7%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Una conclusione a cui sono arrivato è che il valore exp_rate costante non rende a causa del numero elevatissimo di possibili stati.
Con 200k round di training ci sono circa 29milioni di entries.

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

test con 200k round, exp_rate 0.5 decrescente (-0.1) ogni 40k round:
Win rate: 7.82%
Lose rate: 2.18%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Credo che il test con due reinforcement players che giocano uno contro l'altro sia stato male interpretato in quanto, nella fase di test, i due giocatori causavano un loop, è quindi necessario implementare il controllo di loop aggiungendo un valore di ritorno pari a -1 in caso di un numero di mosse > 100.

Inoltre credo di riuscire a snellire ulteriormente la policy e il valore di value_dictionary, hashando ogni stato in un valore che va da 0 a 3^25 (forse) che sono 12 cifre, a differenza delle 25 cifre + i segni che ho ora.

Quindi per la prossima versione modifico prima la funzione di hash e poi implemento il controllo di loop.
Inoltre va implementato il controllo per dare la vittoria all'altro giocatore se il giocatore attuale con l'ultima mossa ha causato la vittoria di entrambi.
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

implementato il controllo di loop
aggiunta la funzione che controlla se il giocatore attuale ha causato la vittoria di entrambi (pushato dalla repo del prof)

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

la mia intuizione era corretta, infatti facendo allenare due RL uno contro l'altro per appena 500 round, il test genera un draw rate del 100%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

per quanto riguarda la funzione di hash voglio rimappare tutto quanto, eliminare i valori negativi e ridurre significativamente il numero di cifre.

ad esempio:
-1  -> 0
0   -> 1
1   -> 2

valore minimo 
0000000000000000000000000 -> 0

valore massimo
2222222222222222222222222 -> 847288609442

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

bugia, credo che rimappare tutto e calcolare una funzione che trasformi da base 3 a base 10 o 16 sia troppo costoso e mi farebbe risparmiare qualche megabyte.
A questo punto credo che il problema sia il fatto che, essendoci moltissimi stati, vengono salvati anche degli stati che hanno valore 0 e che quindi non sono mai stati visitati durante il training.
Bisogna fare in modo che il valore 0 sia assegnato solo agli stati che vengono effettivamente visitati, in quanto vengono aggiunti anche quelli utilizzati solo per verificare se una determinata mossa è possibile.

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

test dopo aver implementato la riduzione del numero degli stati.
500 round RL vs Random:
Prima:  14.8MB, 231869 entries, win rate: 25%
Dopo:   1.3 MB, 20693  entries, win rate: 27%

Test effettuato con successo, il numero di entries si è ridotto notevolmente, seguono test.

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Test 100k round, RL_Player_1 VS Random_Player_2, ogni 20k round salva la policy e la valuta printando un grafico basato sul numero di vittorie.

exp_rate = 0.3 (fixed)

20k round:
Entries: 715637
Win rate: 0.33%
Lose rate: 0.66%
Draw rate: 0.01%

40k round:
Entries: 1366138
Win rate: 0.4%
Lose rate: 0.6%
Draw rate: 0.0%

60k round:
Entries: 1972660
Win rate: 0.55%
Lose rate: 0.44%
Draw rate: 0.01%

80k round:
Entries: 2548963
Win rate: 0.64%
Lose rate: 0.35%
Draw rate: 0.01%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Test 100k round, RL_Player_1 VS Random_Player_2, ogni 20k round salva la policy e la valuta printando un grafico basato sul numero di vittorie.

exp_rate = 0.4 -> 0.0 (decreasing)

20k round, exp_rate = 0.4:
Entries: 656974
Win rate: 0.36%
Lose rate: 0.6%

40k round, exp_rate = 0.3:
Entries: 1144848
Win rate: 0.56%
Lose rate: 0.43%
Draw rate: 0.01%

60k round, exp_rate = 0.2:
Entries: 1445892
Win rate: 0.65%
Lose rate: 0.33%
Draw rate: 0.02%

80k round, exp_rate = 0.1:
Entries: 1703642
Win rate: 0.67%
Lose rate: 0.33%
Draw rate: 0.0%

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TODO: 
- test facendo giocare due RL players uno contro l'altro con e senza diminuzione di exp_rate
- dare un reward inversamente proporzionale a exp_rate
- dare un reward basato sul numero di step utilizzati per vincere (trajectory size media 28)

N.B. in teoria le due policies dei RL players sono le stesse ma opposte quindi ne basta una in cui il primo player cerca il valore più alto nelle possible_moves e il secondo il valore più basso.

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Tabella comparativa per capire quali sono i migliori parametri da utilizzare per il training:
round:      1k, 2k, 5k, 10k
players:    RL vs Random, RL vs RL
exp_rate:   0.5, 0.3
decreasing: yes, no
reward:     [+1, -1], [+1, -1]-[step/100]

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

dopo ulteriori test mi sono reso conto che i valori ottenuti non sono attendibili, il numero di partite di test è troppo basso, deve essere almeno 1k.
Inoltre il numero di entries è ancora troppo grande e non riesco a capire da cosa dipenda.

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Prossimi miglioramenti:
- Implementare un controllo delle simmetrie (come in TicTacToe)
- Implementare un controllo sugli stati già visitati durante la partita per evitare loop infiniti
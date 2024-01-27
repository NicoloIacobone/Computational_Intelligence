This is the new version of the project.
It does not have new features with respect to the old one, but it was needed (I didn't know) to not modify any file given by the teacher.
I created a separated my_players.py file to put my players in it.
I also created a new file called my_game.py that extends the original game.py in order to use private methods like __move.

The first problem faced consists in three different commits that are legal to present for the exam.
The problem is that two of them seems to work normally, the third one doesn't.
In particular, training the same RL agent with a Random agent, 1000 episodes, 0.3 exp_rate gives different results with one version, that outperforms the other two: the agent wins 33% of the games, while the other two versions win 18% of the games.

The first thing I want to try is a different way of hashing the state:
old version: hashable_state = np.array2string(game._board.flatten(), separator = '') 
new version: hashable_state = str(test_board._board.flatten())

The test is made with 5k rounds, RL vs Random, 0.3 exp_rate. I will compare the execution time and the size of the policy file.
old_hash_function: 1:36 min, 5.97MB
new_hash_function: 1:41 min, 8.66MB

I will continue with the old_hash_function.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

From this moment I will use the last commit, 0edba49 "revert to 6th of january"

------------------------------------------------------------------------------------------------------------------------------------------------------------

Let's apply some modifications to the reward distribution:
<!-- - [ ] +1 reward if it changes the board (i.e. if it is a move that reduces by 1 the number of -1 (neutral) cells) -->
- [X] +10 reward if it wins the game
- [X] -10 reward if it loses the game
- [X] (lose_reward * 1000) to the move that makes the opponent win the game, no reward to the other moves (can be done with an inner function that checks if the move makes the opponent win)
- [X] (win_reward * 1000) to the move that makes the agent win the game
- [X] -abs(reward / 50) reward for each move (to improve faster wins)

Some more modifications to the make_move function:
- [] avoid loops (i.e. avoid moves that make the board return to a state that is already in the trajectory)

I have a doubt: why to hash the states ? Can't I simply use the board as a key ? I will try to do so.
It's not possible due to the fact that the key of a dictionary, in Python, must be hashable. I tried using tuple(test_board._board.flatten()) and it work, and it is twice faster than the np.array2string function. Love it!

------------------------------------------------------------------------------------------------------------------------------------------------------------

<!-- Everything seems working better, I have modified a little bit the reward distribution, but I've noticed that the backpropagation is not working properly.
The fact is that the states can be both responsible of winning and losing states.
There is the need to count how many times a move is responsible of a win and how many times it is responsible of a loss using a counter. -->

------------------------------------------------------------------------------------------------------------------------------------------------------------

Trying next training using the following parameters:
- learning_rate = 0.1, halved every 1/5 of the episodes
- exp_rate = 0.9, halved every 1/5 of the episodes
- win_reward = +10
- lose_reward = -10
- step_penalty = 1% of the win_reward

------------------------------------------------------------------------------------------------------------------------------------------------------------

Test RL vs Random, 500k episodes with parameters as above:


------------------------------------------------------------------------------------------------------------------------------------------------------------

Starting implementing minimax agent.

------------------------------------------------------------------------------------------------------------------------------------------------------------


Per il nico del futuro: guarda il file policy_1.txt, i primi 5 valori sono troppo alti, devi capire perché e sistemare il codice. È un problema di backpropagation, probabilmente.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Maybe I fixed it. (spoiler, I didn't)

------------------------------------------------------------------------------------------------------------------------------------------------------------

Fixed the problem of choosing a random move if no state with the available moves is found in the policy. Before this fix the agent always chose the first available move, which is not good. Now I have 50% of winning if it's not trained.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Now I just do some basic training and then I will try to implement a more complex reward function, based on the number of adjacent pieces of the same color.

------------------------------------------------------------------------------------------------------------------------------------------------------------

The training gave again the same result as before, no improvement.
Starting to implement the minimax based agent

------------------------------------------------------------------------------------------------------------------------------------------------------------

Tests with reinforcement learning vs random agent. 100k + 500k + 100k + 3kk episodes. All done starting from 0.9 exp rate and decreasing exp rate that get halved every 20% of the episodes:
Win rate player 1: 85.1%
Lose rate player 1: 14.899999999999999%
Draw rate: 0.0%
Average trajectory size: 16.0
Entries: 43456202
Policy size: 2,69GB

+1kk episodes:
Win rate player 1: 86.6%
Lose rate player 1: 13.4%
Draw rate: 0.0%
Average trajectory size: 49.0
Entries: 54060575
Policy size: 3.35 GB


------------------------------------------------------------------------------------------------------------------------------------------------------------
eval_function = 1
A simple reward function (win = +1, lose = -1, draw = 0)

depth = 3:
Win rate player 1: 86%
Lose rate player 1: 14%

------------------------------------------------------------------------------------------------------------------------------------------------------------
eval_function = 2
A reward function based on the number of pieces on the same row/column/diagonal

depth = 3:
when playing first
Win rate player 1: 86 %
Lose rate player 1: 14 %

when playing second
Win rate player 2: 79%
Lose rate player 2: 21%

depth = 4:

when playing first
Win rate player 1: 99%
Lose rate player 1: 1%

------------------------------------------------------------------------------------------------------------------------------------------------------------

eval_function = 3
A reward function based on the number of pieces on the same row/column/diagonal, with a bonus for a faster win

depth = 3
Win rate player 1: 95%
Lose rate player 1: 5 %

Win rate player 2: 95 %
Lose rate player 2: 5 %

depth = 4
Win rate player 1: 99%
Lose rate player 1: 1%

Win rate player 2: 98%
Lose rate player 2: 2%

------------------------------------------------------------------------------------------------------------------------------------------------------------

eval_function = 4
A reward function based on the number of pieces on the same row/column/diagonal, with a bonus for a faster win and that takes into account the number of pieces owned by the player

depth = 3
Win rate player 1: 96%
Lose rate player 1: 4 %

Win rate player 2: 90%
Lose rate player 2: 10 %

depth = 4
Win rate player 1: 96%
Lose rate player 1: 4%

Win rate player 2: 95%
Lose rate player 2: 5%

------------------------------------------------------------------------------------------------------------------------------------------------------------

Minimax agent is nice, I obtained good win rates. 
It is too slow, I need to find a way to speed it up.

------------------------------------------------------------------------------------------------------------------------------------------------------------

When calling the function to compute the available moves i then iterate over all moves, apply them and call the evaluation function. When applying the move inside the for loop I basically do a repetition of what is being done in the compute available moves function, except for the fact that the old board is restored. I can create a single function that, when the move is valid, applies it and returns the new board. 

before: 4:30min
after : 07:46min

I'll keep the old version.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Tried another approach for minimax player, now it works. Even with the easiest eval function it pefrorms 100% win against random agent, both playing first and second.
(It was a lie, it was setted the fourth eval function)

------------------------------------------------------------------------------------------------------------------------------------------------------------

Up to now I will only calculate which minimax is faster to reach a winning state, I will calculate the average lenght of a match over 100 matches, with depth = 3.

eval_function = 1 (simple reward function)
playing first: 11.7
playing second: 21.04

------------------------------------------------------------------------------------------------------------------------------------------------------------

Starting testing the minimax agent with depth = 3:

Eval function: 0
Player playing FIRST
Minimax won 82 games out of 100
Average game length: 42.01

Eval function: 0
Player playing SECOND
Minimax won 70 games out of 100
Average game length: 52.41

------------------------------------------------------------------------

Eval function: 1
Player playing FIRST
Minimax won 100 games out of 100
Average game length: 14.09

Eval function: 1
Player playing SECOND
Minimax won 99 games out of 100
Average game length: 16.48

------------------------------------------------------------------------

Eval function: 2
Player playing FIRST
Minimax won 100 games out of 100
Average game length: 12.27

Eval function: 2
Player playing SECOND
Minimax won 100 games out of 100
Average game length: 15.55

------------------------------------------------------------------------

Eval function: 3
Player playing FIRST
Minimax won 100 games out of 100
Average game length: 12.01

Eval function: 3
Player playing SECOND
Minimax won 99 games out of 100
Average game length: 16.02

------------------------------------------------------------------------

Eval function: 4
Player playing FIRST
Minimax won 100 games out of 100
Average game length: 11.28

Eval function: 4
Player playing SECOND
Minimax won 100 games out of 100
Average game length: 13.36

------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------

In the meantime I continue training the reinforcement player:
results after 4kk more training episodes:
Win rate player 1: 92.10000000000001%
Lose rate player 1: 7.9%
Draw rate: 0.0%
Average trajectory size: 32.0
Entries: 94777382
Policy size: 5.88 GB

------------------------------------------------------------------------------------------------------------------------------------------------------------
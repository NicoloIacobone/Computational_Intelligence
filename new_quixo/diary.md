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

I'm cleaning a bit the code.
I'm implementing an easier way a human player can play.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Finished to implement the graphical interface for the human player.
Starting implementing the reinforcement player playing second.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Finished to implement the reinforcement player playing second.
Restored main.py and game.py to the original version.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Starting implementing symmetric states.

"Rotating or mirroring the board does not change the state value. There fore states can be grouped in equivalence classes. Said differently, states with symmetrical boards can be merged into a single node in the game graph. This optimization divides approximately by eight the number of states: four being due to rotations, and two to vertical mirroring. Note that the horizontal mirroring is the same as vertical mirroring and 180° rotation. Also, swapping the active player and ﬂipping all Xs and Os to Os and Xs respectively creates a new equivalent state. Figure 2 illustrates these notions. All four states are equivalent."

------------------------------------------------------------------------------------------------------------------------------------------------------------

Started testing the symmetry implementation.
In the meanwhile, I discovered I could save half of the space saving the hashed state as board.astype(np.int8).flatten().tobytes().
Before, each element of the board was saved using 2 bytes, now it is saved using 1 byte.

To save some resources I can store the not-hashed version of the boards in the trajectory.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Finished implementing the symmetry. Now I need to implement the mirroring.
- [ ] To save some resources I can store the not-hashed version of the boards in the trajectory.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Applied the mirroring.
There is some bugs, over 150k states, there are 200 symmetries, idk why

------------------------------------------------------------------------------------------------------------------------------------------------------------

finished bug fixing, now it works.
testing the size of the policy file with 10k episodes
new size:       5MB
new entries:    134297

-------------------------
old size:       8.5MB
old entries:    137146

------------------------------------------------------------------------------------------------------------------------------------------------------------

There's something wrong, I have no decreasing in the number of entries, it should be halved.

------------------------------------------------------------------------------------------------------------------------------------------------------------

With 3 rounds of 1kk training it works:
Win rate player 1: 91.60000000000001%
Lose rate player 1: 8.4%
Average trajectory size: 9.884
Entries: 10238608
Policy size: 356MB

For the same win rate as before I got around:
20 times less entries
20 times less policy size
2,5 times less training rounds to obtain the same win rate

------------------------------------------------------------------------------------------------------------------------------------------------------------

+1kk train episodes on player_1:
Win rate player 1: 92.0%
Lose rate player 1: 8.0%
Entries: 11639934
policy size: 430MB

+1kk train episodes on player_1:
Win rate player 1: 93.4%
Lose rate player 1: 6.6000000000000005%
Entries: 13477063
policy size: 498MB

------------------------------------------------------------------------------------------------------------------------------------------------------------

1kk train episodes on player_2:
Win rate player 2: 85.1%
Lose rate player 2: 14.899999999999999%
Average trajectory size: 5.0
Entries: 4380294
Policy size: 162MB

+1kk train episodes on player_2:
Win rate player 2: 87.9%
Lose rate player 2: 12.1%
Entries: 7404710
policy size: 274MB

------------------------------------------------------------------------------------------------------------------------------------------------------------

Comment all the lines in all files:
- [X] utils.py
- [X] my_players.py
- [X] my_game.py

------------------------------------------------------------------------------------------------------------------------------------------------------------

GitHub only allows to upload files up to 100MB. I thought to link the policy file from Google Drive, but the teacher said it's not a good idea. I need to find a way to let the policy file smaller.

Maybe it is possible to hash the states using less than 25*8its per board.

The board has 25 values that can be -1, 0, 1. I can use 2 bits for each value, so I can use 50 bits instead of 200 bits.

Hashing the board after mapping the values in their "binary" representation halved the size of the policy file, but also halved the execution speed.

It is also possible to compress the policy file using the gzip library.

After compressing with gzip, the policy file is less than 1MB. Trying to compress the original file of 500MB, it becomes 130MB. I will use this approach.

------------------------------------------------------------------------------------------------------------------------------------------------------------

New size (gzip + new_hash): 971kb
New entries:                133009

Old size:                   5MB
Old entries:                134297

------------------------------------------------------------------------------------------------------------------------------------------------------------

I started the training during the night, every million training matches it should've taken 7 hours but in the morning, after 7hr45mins it had only made 383k training matches. I attach the results:

Win rate player 1: 85.5%
Lose rate player 1: 14.499999999999998%
Average trajectory size: 11.458
Entries: 2321057
Policy size: 18MB

+100k training matches:
Win rate player 1: 86.9%
Lose rate player 1: 13.100000000000001%
Average trajectory size: 11.077
Entries: 4762877
Policy size: 35MB

+50k training matches:
Win rate player 1: 89.0%
Lose rate player 1: 11.0%
Average trajectory size: 10.686
Entries: 5025361
Policy size: 37,8MB

---------------------------------------------
Win rate player 2: 80.7%
Lose rate player 2: 19.3%
Average trajectory size: 12.927
Entries: 2623850
Policy size: 20MB

+100k training matches:
Win rate player 2: 82.5%
Lose rate player 2: 17.5%
Average trajectory size: 11.845
Entries: 4908261
Policy size: 37,6MB

+200k training matches:
Win rate player 2: 83.2%
Lose rate player 2: 16.8%
Average trajectory size: 16.0
Enries: 7367171
Policy size: 56,6MB

------------------------------------------------------------------------------------------------------------------------------------------------------------

On average:
Entries per MB before: 27k
Entries per MB after:  130k

It is 4.8 times better.

------------------------------------------------------------------------------------------------------------------------------------------------------------

To divide the policy file I can separately store the keys and the values, and then load them separately.

------------------------------------------------------------------------------------------------------------------------------------------------------------

Done, it works. Now I have a keys file of 83,6MB and a values file of 21,9MB. Before the total file was 130MB

------------------------------------------------------------------------------------------------------------------------------------------------------------
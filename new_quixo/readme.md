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

Maybe I fixed it.
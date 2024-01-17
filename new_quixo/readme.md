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
-------------------------------------------------------------------------------------------------------------------------------------------------------------

Let's apply some modifications to the reward distribution:
<!-- - [ ] +1 reward if it changes the board (i.e. if it is a move that reduces by 1 the number of -1 (neutral) cells) -->
- [ ] +10 reward if it wins the game
- [ ] -10 reward if it loses the game
- [ ] (lose_reward * 1000) to the move that makes the opponent win the game, no reward to the other moves (can be done with an inner function that checks if the move makes the opponent win)
- [ ] -abs(reward / 50) reward for each move (to improve faster wins)

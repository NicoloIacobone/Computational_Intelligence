The aim of this project is to implement an agent able to play the game `Quixo`.
I implemented two different agents:
- `ReinforcementPlayer`: an agent that learns how to play the game by playing against a RandomPlayer and updating its policy
- `MinimaxPlayer`: an agent that uses the Minimax algorithm to play the game

I also implemented a `HumanPlayer` that allows a human to play against the agent with an easy-to-use graphical interface.

Just run the `main.py` file to select the players and play the game.

If you want to use ReinforcementPlayer, MinimaxPlayer and HumanPlayer, you need to import them from `my_players.py`

To train the ReinforcementPlayer, you need to firstly import the `Utils` class from `utils.py` and then call the `train` method with the desired parameters:
- First player to move
- Second player to move
- Number of games to play
- Win reward
- Lose reward
- Path to save the policy
- Decrease the exploration rate by 50% every 20% of the games
- Plot the win rate to see how the decreasing exploration rate affects the win rate

To test the ReinforcementPlayer, you need to firstly import the `Utils` class from `utils.py` and then call the `test` method with the desired parameters:
- First player to move
- Second player to move
- Number of games to play
- Path to load the policy

In the MinimaxPlayer it is possible, when instantiating the player, to set the depth of the search tree and the heuristic to use (from 0, the simplest, to 3, the more complex).

More information about the players and their parameters can be found in the respective files, every line of code is commented.

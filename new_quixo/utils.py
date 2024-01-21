''' In this class are defined some useful functions: Train, Test, Evaluate '''
from game import Game, Player
from my_game import MyGame
from my_players import ReinforcementPlayer
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import ceil

class Utils:
    def __init__(self) -> None:
        pass

    def train(self, 
              player1: Player, # player that moves first
              player2: Player, # player that moves second
              games: int = 1_000, # number of training games
              win_reward: int = 10, # reward for winning
              lose_reward: int = -10, # reward for losing
              policy_name = "test_policy", # name of the policy
              plot: bool = False, # if we want to plot the win rate
              decreasing_exp_rate: bool = False, debug = True) -> None: # if we want to decrease the random move value during training
        
        if isinstance(player1, ReinforcementPlayer):
            player1_reinforcement = True
            player1.training = True
        else:
            player1_reinforcement = False
        if isinstance(player2, ReinforcementPlayer):
            player2_reinforcement = True
            player2.training = True
        else:
            player2_reinforcement = False

        if plot:
            plot_variables = []

        for round in tqdm(range(games)):
            if plot and (round + 1) % (games // 20) == 0: # every 20% of the games
                if decreasing_exp_rate: # if the random move should decrease
                    if player1_reinforcement:
                        player1.set_random_move(player1.random_move * 0.5) # decrease the random move by 50%
                        player1.set_learning_rate(player1.learning_rate * 0.5) # decrease the learning rate by 50%
                    if player2_reinforcement:
                        player2.set_random_move(player2.random_move * 0.5) # decrease the random move by 50%
                        player2.set_learning_rate(player2.learning_rate * 0.5) # decrease the learning rate by 50%
                win_rate = self.test(player1, player2, 500, policy_name, True) # test the player against a random player
                plot_variables.append(win_rate) # append the win rate to the list to plot it later
                    
            game = Game() # create a new game
            win = game.play(player1, player2) # play the game

            if debug:
                suicide = win != game.current_player_idx # if the move of a player caused the winning of the opponent
                if not suicide and win == 0:
                    player1.give_reward(win_reward)
                elif win == 1:
                    if suicide:
                        player1.give_reward(lose_reward, True)
                    else:
                        player1.give_reward(lose_reward)
            else:
                suicide_opponent = win != game.current_player_idx # if the move of a player caused the winning of the opponent
                if win == 0: # if first player won
                    if player1_reinforcement: # if I won (RL agent)
                        if not suicide_opponent: # if I made the winning move
                            player1.give_reward(win_reward) # I receive the positive reward
                            # otherwise if the opponent made me win with his last move, I don't reward the agent (i.e. skip this game)

                    if player2_reinforcement: # if I lose (RL agent)
                        if not suicide_opponent: # if the opponent made the winning move
                            player2.give_reward(lose_reward) # I receive a negative reward
                        else: # if I made myself lose with my last move (suicide)
                            player2.give_reward((lose_reward * 1000), True) # I receive a strong negative reward (only on the last move)

                elif win == 1: # if second player won
                    if player1_reinforcement: # if I lose (RL agent)
                        if not suicide_opponent: # if the opponent made the winning move
                                player1.give_reward(lose_reward) # I receive a negative reward
                        else: # if I made myself lose with my last move (suicide)
                            player1.give_reward((lose_reward * 1000), True) # I receive a strong negative reward (only on the last move)
                    if player2_reinforcement: # if I won (RL agent)
                        if not suicide_opponent: # if I made the winning move
                            player2.give_reward(win_reward) # I receive the positive reward
                        # otherwise if the opponent made me win with his last move, I don't reward the agent (i.e. skip this game)

        # when training finishes
        if player1_reinforcement:
            player1.create_policy(policy_name + "_1")
            player1.training = False
        if player2_reinforcement:
            player2.create_policy(policy_name + "_2")
            player2.training = False

        if plot:
            # create the plot
            plt.plot(plot_variables)
            plt.xlabel("Training games")
            plt.ylabel("Win rate")
            plt.title("Win rate over training games")
            plt.show()

    def test(self, 
             player1: Player, 
             player2: Player, 
             games: int = 1_000, 
             policy_name = "test_policy", 
             plot: bool = False) -> None:
        
        win_rate_player_1 = 0
        lose_rate_player_1 = 0
        draw_rate = 0
        trajectory_size = 0

        if isinstance(player1, ReinforcementPlayer):
            player1_reinforcement = True
        else:
            player1_reinforcement = False

        if isinstance(player2, ReinforcementPlayer):
            player2_reinforcement = True
        else:
            player2_reinforcement = False

        if player1_reinforcement:
            old_random_move_1 = player1.random_move # save the old random move to restore it later
            player1.set_random_move(0) # set the random move to 0 to perform the test
            if not plot:
                player1.load_policy(policy_name + "_1")
        if player2_reinforcement:
            old_random_move_2 = player2.random_move # save the old random move to restore it later
            player2.set_random_move(0) # set the random move to 0 to perform the test
            if not plot:
                player2.load_policy(policy_name + "_2")

        for _ in range(games):
            game = Game()
            win = game.play(player1, player2)
            if win == 0:
                win_rate_player_1 += 1
            elif win == 1:
                lose_rate_player_1 += 1
            else:
                draw_rate += 1
            trajectory_size += len(player1.trajectory) # it is the number of moves done by player 1

        if plot: # if we are plotting the win rate = we have to continue the training
            if player1_reinforcement:
                player1.set_random_move(old_random_move_1) # restore the old random move
            if player2_reinforcement:
                player2.set_random_move(old_random_move_2) # restore the old random move
            return win_rate_player_1 / games * 100 # return the win rate

        print(f"Win rate player 1: {win_rate_player_1 / games * 100}%") # print the win rate
        print(f"Lose rate player 1: {lose_rate_player_1 / games * 100}%") # print the lose rate
        print(f"Draw rate: {draw_rate / games * 100}%") # print the draw rate
        print(f"Average trajectory size: {trajectory_size / games}") # print the average trajectory size
        print("---------------------------------------------") # print a separator

    def evaluate_player(self, player_to_evaluate: ReinforcementPlayer, policy = None):
        if policy is not None:
            player_to_evaluate.load_policy(policy)

        print(f"Entries: {len(player_to_evaluate.value_dictionary)}") # print the number of entries in the dictionary

        if policy is not None:
            print(f"Policy size: {player_to_evaluate.get_policy_size(policy)/1_000_000:.2f} MB") # print the size of the policy in MB
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
              policy_name = "policies/policy", # name of the policy
              plot: bool = False, # if we want to plot the win rate
              decreasing_exp_rate: bool = False) -> None: # if we want to decrease the random move value during training
        
        if isinstance(player1, ReinforcementPlayer): # if player 1 is a reinforcement player
            player1_reinforcement = True # set the flag to True
            player1.training = True # set the training flag to True to not let it load the policy nor update the trajectory
        else:
            player1_reinforcement = False
        if isinstance(player2, ReinforcementPlayer): # if player 2 is a reinforcement player
            player2_reinforcement = True # set the flag to True
            player2.training = True # set the training flag to True to not let it load the policy nor update the trajectory
        else:
            player2_reinforcement = False

        if plot: # if we are plotting the win rate
            plot_variables = [] # create a list to store the win rate values

        for round in tqdm(range(games)):
            if decreasing_exp_rate or plot: # if we want to decrease the random move value during training or we are plotting the win rate
                if (round + 1) % (games // 20) == 0: # every 20% of the games
                    if decreasing_exp_rate: # if the random move should decrease
                        if player1_reinforcement:
                            player1.set_random_move(player1.random_move * 0.5) # decrease the random move by 50%
                            player1.set_learning_rate(player1.learning_rate * 0.5) # decrease the learning rate by 50%
                        if player2_reinforcement:
                            player2.set_random_move(player2.random_move * 0.5) # decrease the random move by 50%
                            player2.set_learning_rate(player2.learning_rate * 0.5) # decrease the learning rate by 50%
                    if plot:
                        win_rate = self.test(player1, player2, 500, policy_name, True) # test the player against a random player
                        plot_variables.append(win_rate) # append the win rate to the list to plot it later
                    
            game = Game() # create a new game
            win = game.play(player1, player2) # play the game

            suicide = win != game.current_player_idx # if the move of a player caused the winning of the opponent
            if win == 0: # player X won
                if suicide: # if player X won because player O made a mistake
                    # player X does not receive any reward
                    if player2_reinforcement:
                        player2.give_reward(lose_reward, True) # player O receives a huge negative reward for the mistake
                else: # if X won because of the right move it made
                    if player1_reinforcement:
                        player1.give_reward(win_reward) # player X takes the winning reward
                    if player2_reinforcement:
                        player2.give_reward(lose_reward) # player O takes the losing reward
            elif win == 1: # player O won
                if suicide: # if player O won because player X made a mistake
                    # player O does not receive any reward
                    if player1_reinforcement:
                        player1.give_reward(lose_reward, True) # player X receives a huge negative reward for the mistake
                else: # if O won because of the right move it made
                    if player2_reinforcement:
                        player2.give_reward(win_reward)
                    if player1_reinforcement:
                        player1.give_reward(lose_reward)

        # when training finishes
        if player1_reinforcement:
            player1.create_policy(policy_name + "_1")
            player1.training = False # set the training flag to false so it can be used for testing purposes
        if player2_reinforcement:
            player2.create_policy(policy_name + "_2")
            player2.training = False

        if plot:
            plt.plot(plot_variables)
            plt.xlabel("Training games")
            plt.ylabel("Win rate")
            plt.title("Win rate over training games")
            plt.show()

    def test(self, 
             player1: Player, # player that moves first
             player2: Player, # player that moves second
             games: int = 1_000, # number of test games
             policy_name = "policies/policy", # name of the policy
             plot: bool = False) -> None: # if we want to plot the win rate
        
        win_rate_player_1 = 0 # count of won games for player 1
        win_rate_player_2 = 0 # count of won games for player 2
        lose_rate_player_1 = 0 # count of lost games for player 1
        lose_rate_player_2 = 0 # count of lost games for player 2
        draw_rate = 0 # count of draw games
        trajectory_size_1 = 0 # sum of the trajectory size of player 1
        trajectory_size_2 = 0 # sum of the trajectory size of player 2

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
            if not plot: # if we are not plotting the win rate we have to load the policy because this is the only time we use it
                player1.load_policy(policy_name + "_1")
        if player2_reinforcement:
            old_random_move_2 = player2.random_move # save the old random move to restore it later
            player2.set_random_move(0) # set the random move to 0 to perform the test
            if not plot: # if we are not plotting the win rate we have to load the policy because this is the only time we use it
                player2.load_policy(policy_name + "_2")

        for _ in range(games):
            game = Game()
            win = game.play(player1, player2) # play the game
            if win == 0: # if player 1 won
                win_rate_player_1 += 1 # increase the win rate of player 1
                lose_rate_player_2 += 1 # increase the lose rate of player 2
            elif win == 1: # if player 2 won
                lose_rate_player_1 += 1 # increase the lose rate of player 1
                win_rate_player_2 += 1 # increase the win rate of player 2
            else:
                draw_rate += 1
            if player1_reinforcement:
                trajectory_size_1 += len(player1.trajectory) # add the trajectory size to the sum
            if player2_reinforcement:
                trajectory_size_2 += len(player2.trajectory) # add the trajectory size to the sum

        if plot: # if we are plotting the win rate = we have to continue the training
            if player1_reinforcement:
                player1.set_random_move(old_random_move_1) # restore the old random move
            if player2_reinforcement:
                player2.set_random_move(old_random_move_2) # restore the old random move
            return win_rate_player_1 / games * 100 # return the win rate

        if player1_reinforcement:
            print(f"Win rate player 1: {win_rate_player_1 / games * 100}%") # print the win rate
            print(f"Lose rate player 1: {lose_rate_player_1 / games * 100}%") # print the lose rate
            print(f"Draw rate: {draw_rate / games * 100}%") # print the draw rate
            print(f"Average trajectory size: {trajectory_size_1 / games}") # print the average trajectory size
            print("---------------------------------------------") # print a separator
        if player2_reinforcement:
            print(f"Win rate player 2: {win_rate_player_2 / games * 100}%")
            print(f"Lose rate player 2: {lose_rate_player_2 / games * 100}%")
            print(f"Draw rate: {draw_rate / games * 100}%")
            print(f"Average trajectory size: {trajectory_size_2 / games}")
            print("---------------------------------------------")

    def evaluate_player(self, player_to_evaluate: ReinforcementPlayer, policy = None):
        if policy is not None:
            player_to_evaluate.load_policy(policy)

        print(f"Entries: {len(player_to_evaluate.value_dictionary)}") # print the number of entries in the dictionary

        if policy is not None:
            policy_size = player_to_evaluate.get_policy_size(policy)
            if policy_size >= 1_000_000_000:
                print(f"Policy size: {policy_size/1_000_000_000:.2f} GB") # print the size of the policy in GB
            else:
                print(f"Policy size: {policy_size/1_000_000_000:.2f} GB") # print the size of the policy in MB
            
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
              player1: Player, 
              player2: Player, 
              games: int = 1_000, 
              win_reward: int = 1, 
              lose_reward: int = -1, 
              policy_name = "test_policy", 
              plot: bool = False, 
              decreasing_exp_rate: bool = False,
              new_reward: bool = False) -> None:
        
        if isinstance(player1, ReinforcementPlayer):
            player1.training = True
        if isinstance(player2, ReinforcementPlayer):
            player2.training = True

        if plot:
            plot_variables = []

        for round in tqdm(range(games)): 
            if plot and (round + 1) % (games // 10) == 0: # every 10% of the games
                if decreasing_exp_rate: # if the random move should decrease
                    if isinstance(player1, ReinforcementPlayer):
                        player1.set_random_move(player1.random_move * 0.9) # decrease the random move by 10%
                    if isinstance(player2, ReinforcementPlayer):
                        player2.set_random_move(player2.random_move * 0.9)
                win_rate = self.test(player1, player2, 500, policy_name, True) # test the player against a random player
                if plot:
                    plot_variables.append(win_rate) # append the win rate to the list to plot it later
                    
            game = Game()
            win = game.play(player1, player2)
            if win == 0:
                if isinstance(player1, ReinforcementPlayer):
                    player1.give_reward(win_reward)
                    # player1.give_reward(win_reward - (ceil(len(game.trajectory_player_1) / 100)), game.trajectory_player_1)
                if isinstance(player2, ReinforcementPlayer):
                    player2.give_reward(lose_reward - (ceil(len(player2.trajectory) / 100)))
            elif win == 1:
                if isinstance(player1, ReinforcementPlayer):
                    if not new_reward:
                        player1.give_reward(lose_reward)
                    else:
                        player1.give_reward(lose_reward - (ceil(len(player1.trajectory) / 100)))
                if isinstance(player2, ReinforcementPlayer):
                    player2.give_reward(win_reward - (ceil(len(player2.trajectory) / 100)))

        if isinstance(player1, ReinforcementPlayer):
            player1.create_policy(policy_name + "_1")
            player1.training = False
        if isinstance(player2, ReinforcementPlayer):
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
            old_random_move_1 = player1.random_move # save the old random move to restore it later
            player1.set_random_move(0) # set the random move to 0 to perform the test
            if not plot:
                player1.load_policy(policy_name + "_1")
        if isinstance(player2, ReinforcementPlayer):
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
            if isinstance(player1, ReinforcementPlayer):
                player1.set_random_move(old_random_move_1) # restore the old random move
            if isinstance(player2, ReinforcementPlayer):
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
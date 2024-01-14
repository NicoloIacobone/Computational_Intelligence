from game import Game, Move, Player
from main import RandomPlayer, ReinforcementPlayer
from tqdm import tqdm
import matplotlib.pyplot as plt

''' In this class are defined some useful functions:
    Train, Test, Evaluate '''

class Utils:
    def __init__(self) -> None:
        pass

    def train(self, 
              player1: Player = RandomPlayer(), 
              player2: Player = RandomPlayer(), 
              games: int = 1_000, 
              win_reward: int = 1, 
              lose_reward: int = -1, 
              policy_name = "test_policy", 
              plot: bool = False, 
              decreasing_exp_rate: bool = False) -> None:
        
        if plot:
            plot_variables = []

        for round in tqdm(range(games)):
            if plot and (round + 1) % (games // 10) == 0:
                if decreasing_exp_rate:
                    if isinstance(player1, ReinforcementPlayer):
                        player1.set_random_move(player1.random_move * 0.9)
                    if isinstance(player2, ReinforcementPlayer):
                        player2.set_random_move(player2.random_move * 0.9)
                win_rate = self.test(player1, player2, 500, policy_name, True)
                plot_variables.append(win_rate)

            game = Game()
            win = game.play(player1, player2)
            if win == 0:
                if isinstance(player1, ReinforcementPlayer):
                    player1.give_reward(win_reward, game.trajectory)
                if isinstance(player2, ReinforcementPlayer):
                    player2.give_reward(lose_reward, game.trajectory)
            elif win == 1:
                if isinstance(player1, ReinforcementPlayer):
                    player1.give_reward(lose_reward, game.trajectory)
                if isinstance(player2, ReinforcementPlayer):
                    player2.give_reward(win_reward, game.trajectory)

        if isinstance(player1, ReinforcementPlayer):
            player1.create_policy(policy_name + "_1")
        if isinstance(player2, ReinforcementPlayer):
            player2.create_policy(policy_name + "_2")

        if plot:
            # create the plot
            plt.plot(plot_variables)
            plt.xlabel("Training games")
            plt.ylabel("Win rate")
            plt.title("Win rate over training games")
            plt.show()

    def test(self, 
             player1: Player = RandomPlayer(), 
             player2: Player = RandomPlayer(), 
             games: int = 1_000, 
             policy_name = "test_policy", 
             plot: bool = False) -> None:
        
        win_rate_player_1 = 0
        lose_rate_player_1 = 0
        draw_rate = 0
        trajectory_size = 0

        if isinstance(player1, ReinforcementPlayer):
            old_random_move_1 = player1.random_move
            player1.set_random_move(0)
            if not plot:
                player1.load_policy(policy_name + "_1")
        if isinstance(player2, ReinforcementPlayer):
            old_random_move_2 = player2.random_move
            player2.set_random_move(0)
            if not plot:
                player2.load_policy(policy_name + "_2")

        for _ in tqdm(range(games)):
            game = Game()
            win = game.play(player1, player2)
            if win == 0:
                win_rate_player_1 += 1
            elif win == 1:
                lose_rate_player_1 += 1
            else:
                draw_rate += 1
            trajectory_size += len(game.trajectory)

        if plot:
            if isinstance(player1, ReinforcementPlayer):
                player1.set_random_move(old_random_move_1)
            if isinstance(player2, ReinforcementPlayer):
                player2.set_random_move(old_random_move_2)
            return win_rate_player_1

        print(f"Win rate player 1: {win_rate_player_1 / games * 100}%")
        print(f"Lose rate player 1: {lose_rate_player_1 / games * 100}%")
        print(f"Draw rate: {draw_rate / games * 100}%")
        print(f"Average trajectory size: {trajectory_size / games}")
        print("---------------------------------------------")

    def evaluate_player(self, player_to_evaluate: ReinforcementPlayer, policy = None):
        if policy is not None:
            player_to_evaluate.load_policy(policy)

        print(f"Entries: {len(player_to_evaluate.value_dictionary)}")
        if policy is not None:
            print(f"Policy size: {player_to_evaluate.get_policy_size(policy)/1_000_000:.2f} MB")

from agent import Agent
from gameChess import chess_game



def QLearn(game, agent):
  game.reset()
  agent.n_games += 1
  agent.train_long_memory()
  print(f"Game {agent.n_games} ended!")

def train():
    agent = Agent()
    game = chess_game()

    # set games if already gaming
    while True:
        """
        ----------------------------------------------------------------
        WHITE'S TURN
        ----------------------------------------------------------------
        """
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(game.board, "white")

        # perform move and get new state
        reward_white, done_white = game.step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward_white, state_new, done_white)

        # remember
        agent.remember(state_old, final_move, reward_white, state_new, done_white)

        """
        ----------------------------------------------------------------
        BLACK'S TURN
        ----------------------------------------------------------------
        """
        if done_white:
            QLearn(game, agent)
            agent.saveModel('model.pth')
            continue

        state_old = state_new

        # get move
        final_move = agent.get_action(game.board, "black")

        # perform move and get new state
        reward_black, done_black = game.step(final_move)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward_black, state_new, done_black)

        # remember
        agent.remember(state_old, final_move, reward_black, state_new, done_black)

        if done_black:
            QLearn(game, agent)
            agent.saveModel('model.pth')


if __name__ == '__main__':
    train()
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
        final_move, rand = agent.get_action(game.board, "white")

        # perform move and get new state
        reward_white, done_white = game.step(final_move)
        try:
            # adjust rewards
            reward_black += reward_white
            agent.remember(state_old_black, rand_black, reward_black, state_new_black, done_black)
        except UnboundLocalError:
            pass
        state_new = game.get_state()
        """
        ----------------------------------------------------------------
        BLACK'S TURN
        ----------------------------------------------------------------
        """
        if done_white:  # white wins or draw
            QLearn(game, agent)
            agent.saveModel('model.pth')
            
            if reward_white >= game.CHECKMATE:  # more means win (win reward always more than 200 else i stoopid)
                agent.memory[-2][2] += game.CHECKMATE
            continue
        else:
            # train short memory only IF it isnt checkmate
            try:
                agent.train_short_memory(state_old_black, rand_black, reward_black, state_new_black, done_black)
            except UnboundLocalError:
                pass

        state_old_black = state_new

        # get move
        final_move_black, rand_black = agent.get_action(game.board, "black")

        # perform move and get new state
        reward_black, done_black = game.step(final_move_black)
        reward_black *= -1
        state_new_black = game.get_state()
        
        # deduct for black
        reward_white -= reward_black
        # white remember
        agent.remember(state_old, rand, reward_white, state_new, done_white)
        
        

        if done_black:
            QLearn(game, agent)
            agent.saveModel('model.pth')
            
            if reward_black >= game.CHECKMATE:  # more means win (win reward always more than 200 else i stoopid)
                agent.memory[-2][2] -= game.CHECKMATE
        else:
            # train short memory
            # UnboundedLocalError wont happen because already assigned earlier
            agent.train_short_memory(state_old, rand, reward_white, state_new, done_white)


if __name__ == '__main__':
    train()

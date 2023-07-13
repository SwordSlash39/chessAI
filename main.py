from agent import Agent
from gameChess import chess_game

print()
def QLearn(game, agent):
    print("\033[A                                      \033[A")
    print(f"Game ended! Training...")
    game.reset()
    agent.n_games += 1
    agent.train_long_memory()
    print("\033[A                                      \033[A")
    print("\033[A                                      \033[A")
    print(f"Game {agent.n_games} ended!\n")
    if agent.n_games % 3 == 0:
        print(f"Saving model with {agent.n_games} Games played...")
        agent.saveModel('model.pth')
        print()

def train():
    move = 1
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
            # adjust rewards ADDING IS NOT A BUGG THE EVAL SHOULD BE LOW FOR BLACK
            reward_black += reward_white
            agent.remember(state_old_black, rand_black, reward_black, state_new_black, done_black ,"black")
        except UnboundLocalError:
            pass
        state_new = game.get_state()
        # Add half move
        print ("\033[A                                  \033[A")
        print(f"Half-move {move} of Game {agent.n_games+1} played!")
        move += 1
        """
        ----------------------------------------------------------------
        BLACK'S TURN
        ----------------------------------------------------------------
        """
        if done_white:  # white wins or draw
            QLearn(game, agent)
            move = 0            
            if reward_white >= game.CHECKMATE:  # more means win 
                agent.memory[-2][2] += game.CHECKMATE
            continue
        else:
            # train short memory only IF it isnt checkmate
            try:
                agent.train_short_memory(state_old_black, rand_black, reward_black, state_new_black, done_black, "black")
            except UnboundLocalError:
                pass

        state_old_black = state_new

        # get move
        final_move_black, rand_black = agent.get_action(game.board, "black")

        # perform move and get new state
        reward_black, done_black = game.step(final_move_black)
        reward_black *= -1
        state_new_black = game.get_state()
        
        # new Move!
        print ("\033[A                                  \033[A")
        print(f"Half-move {move} of Game {agent.n_games+1} played!")
        move += 1
        # deduct for black
        reward_white -= reward_black
        # white remember
        agent.remember(state_old, rand, reward_white, state_new, done_white, "white")
        
        

        if done_black:
            QLearn(game, agent)
            move = 0            
            if reward_black >= game.CHECKMATE:  # more means win (win reward always more than 200 else i stoopid)
                agent.memory[-2][2] -= game.CHECKMATE
        else:
            # train short memory
            # UnboundedLocalError wont happen because already assigned earlier
            agent.train_short_memory(state_old, rand, reward_white, state_new, done_white, "white")


if __name__ == '__main__':
    train()

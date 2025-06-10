import torch
import random
import numpy as np
from collections import deque
from main import SnakeGame,Direction,Point

MAX_MEMORY = 100_000# THIS MANY ITEMS WE CAN STORE
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0#parameter to control randomnes
        self.gama = 0#discount rate from the equation
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() - if we  exceed this
        # memory it will automaticly remove elements for use
        # TODO - model, trainer
    def get_state(self, game):
        pass
    def remember(self,state,action,reward,next_state, game_over):
        pass
    def train_long_memory(self):
        pass
    def train_short_memory(self,state,action,reward,next_state, game_over):
        pass
    def get_action(self,state):
        pass
def train():
    plot_scores = []
    plot_mean = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #performe move and get new state
        reward,game_over,score = game.play_step(action=final_move)
        state_new = agent.get_state((game))

        #train the short memory of agent
        agent.train_short_memory(state_old,final_move,reward,state_new, game_over)

        #remember
        agent.remember(state_old,final_move,reward,state_new, game_over)
        if game_over:
            # train the long memory(experience replay)
            game.reset()
            agent.num_games +=1
            agent.train_long_memory()

            if score>record:
                record = score
                # agent.mode.save()
            print('Game',agent.num_games, 'Score',score,'Record',record)
            # TODO - plotting

if __name__=='__main__':
    train()
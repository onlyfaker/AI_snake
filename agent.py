import torch
import random
import numpy as np
from collections import deque
from main import SnakeGame,Direction,Point
from  model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000# THIS MANY ITEMS WE CAN STORE
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0#parameter to control randomnes
        self.gamma = 0.9#discount rate from the equation
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() - if we  exceed this
        # memory it will automatically remove elements for use
        self.model = LinearQNet(11,256,3)
        self.trainer = QTrainer(self.model,learning_rate=LR, gamma=self.gamma)
        # TODO - model, trainer
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x-20, head.y)#-20 bc of the block size
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)


    def remember(self,state,action,reward,next_state, game_over):
        self.memory.append((state,action,reward,next_state, game_over))#popleft if max memory is reached
    def train_long_memory(self):
        if len(self.memory)> BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)#list of tuples
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states, game_overs)

    def train_short_memory(self,state,action,reward,next_state, game_over):
        self.trainer.train_step(state,action,reward,next_state, game_over)
    def get_action(self,state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80-self.num_games
        final_move = [0,0,0]
        if random.randint(0,200)<self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move] =1
        return final_move
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
                agent.model.save()
            print('Game',agent.num_games, 'Score',score,'Record',record)
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.num_games
            plot_mean.append(mean_score)
            plot(plot_scores,plot_mean)
if __name__=='__main__':
    train()
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
        pass
    def get_state(self, game):
        pass
    def remember(self,state,action,reward,next_state, game_over):
        pass
    def train_long_memory(self):
        pass
    def train_short_memory(self):
        pass
    def get_action(self, state):
        pass
def train():
    pass

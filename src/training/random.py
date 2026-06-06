import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
import gym_super_mario_bros
from gymnasium.wrappers import GrayscaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from src.evaluation.MetricLogger import MetricLogger
from src.utils.agents.RandomMario import RandomMario
from src.utils.wrappers import ResizeObservation, SkipFrame

# Initialize Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3', render_mode=None)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=64)

env.observation_space = gym.spaces.Box(
    low=env.observation_space.low,
    high=env.observation_space.high,
    shape=env.observation_space.shape,
    dtype=np.float32
)

env = TransformObservation(env, func=lambda x: x / 255., observation_space=env.observation_space)

state, info = env.reset(seed=42)

save_dir = Path('experiments/logs/random') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = RandomMario(action_dim=env.action_space.n)

logger = MetricLogger(save_dir)

episodes = 1000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    _, info = env.reset(seed=42)

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = mario.act()

        # 5. Agent performs action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 6. Logging
        logger.log_step(reward, None, None)

        # 7. Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=1, #is always random
            step=mario.curr_step
        )
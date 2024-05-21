import gym
from gym import spaces
import numpy as np
import pandas as pd
import shimmy
from stable_baselines3 import DQN


class FirmEnv(gym.Env):
    def __init__(self, data):
        super(FirmEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = spaces.Discrete(4)  # Örneğin, 4 farklı aksiyon
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self.calculate_reward(action)
        next_state = self.data.iloc[self.current_step].values
        return next_state, reward, done, {}

    def calculate_reward(self, action):
        # Ödül hesaplama mantığı
        return np.random.rand()

# Veri yükleme
data = pd.read_csv('qqq.csv')

# Çevreyi oluşturma
env = FirmEnv(data)


# Modeli eğitme
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Tahmin yapma
obs = env.reset()
for i in range(len(data)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break
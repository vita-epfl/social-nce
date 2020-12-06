import gym
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)

    def reset(self, phase='test', test_case=None):
        obs = self.env.reset(phase=phase, test_case=test_case)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return self._frames

    def render(self, mode='human', output_file=None):
        return self.env.render(mode=mode, output_file=output_file)

    def get_global_time(self):
        return self.env.global_time

    def get_human_times(self):
        return self.env.get_human_times()

    def onestep_lookahead(self, action):
        return self.env.onestep_lookahead(action)

def latest_frame(frames, k):
    return frames[-1]
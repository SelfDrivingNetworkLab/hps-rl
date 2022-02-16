import logging

import numpy as np

import gym
from gym.utils import seeding

from .model import Controller

np.set_printoptions(precision=0, suppress=True, linewidth=150)
logger = logging.getLogger(__name__)


class LaserControllerEnv(gym.Env, Controller):
    metadata = {"render.modes": ["human"]}
    """
    Description:
        2D diffractive laser coherent beam combining (CBC) uses a diffractive
        optical element (DOE) to combine a square array of MxM equal amplitude
        beams. The process of CBC is mathematically understood as a 2D
        convolution between a complex input beam array with shape of M by M, and
        the characteristic transmission function of DOE itself of the same
        shape. As a result, the combined beam power is a function of the phase
        of each beam, which need to be actively controlled to maintain optimal
        combining efficiency against environment perturbations.

        This environment enables simulations using two measured DOE
        characteristic transmission functions, for a 3x3 and 9x9 input beam array
        size, respectively. The goal is to maximize combined beam power, or
        normalized combining efficiency, by correcting input beam phases with
        minimal steps, starting from a random phase state.

    Action space:
        Type: Continuous Box(M, M)
        Representing the correction of beam phase array, normalized to [-1, 1]

    Observation space:
        Type: Continuous Box(2M-1, 2M-1)
        Representing the observable far field diffraction intensity pattern,
        where the center is the combined beam.
        May be a larger shape, if considering higher order modes of DOE.

    Reward:
        -1 for every step taken

    Starting State:
        A uniformly distributed random input beam phase array, away from optimal
        value by a `perturb_range` in degrees.

    Episode Termination:
        Normalized efficiency is higher than `done_threshold`, where 1 means
        theroretical maximum combined power.
    """

    def __init__(self, **kwargs):
        super(LaserControllerEnv, self).__init__(**kwargs)
        self.done_threshold = kwargs.get("done_threshold", 0.8)
        self.perturb_range = kwargs.get("perturb_range", 180)
        self.rms_noise = kwargs.get("rms_noise", 0)
        self.viewer = None

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.model.beam_phs.size,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=0, high=self.model.max_pwr, shape=(self.model.pattern.size,), dtype=np.float32
        )
        m = self.model.M // 2 + 1
        self.action_names = [(x, y) for y in range(m - 1, -m, -1) for x in range(-m + 1, m)]
        n = self.model.pattern.shape[-1] // 2 + 1
        self.obs_names = [(x, y) for y in range(n - 1, -n, -1) for x in range(-n + 1, n)]

        logger.info("Action space: {}".format(self.action_space))
        logger.info("Observ space: {}".format(self.observation_space))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ accepts flat input / output array """
        action_2d = action.reshape(self.model.beam_shape)
        self.correct(action_2d * np.pi)  # correct phases in 360 deg

        self.steps += 1
        info = {
            "beam_phase": self.model.beam_phs,
            "combined_power": self.model.combined_power,
            "norm_efficiency": self.model.norm_eta,
        }

        done = bool(info["norm_efficiency"] >= self.done_threshold or self.steps > 500)
        reward = -1
        # reward = self.model.norm_eta - 1

        return self.model.pattern.ravel(), reward, done, info

    def reset(self):
        """ Perturb system state by injecting random phases """
        self.reset_drive()  # zero output
        self.model.reset()  # move to optimal state
        self.model.perturb(self.perturb_range, self.np_random)  # perturb random phases in deg
        self.steps = 0
        return self.model.pattern.ravel()

    def render(self, mode="human"):
        logger.info("Efficiency:{:5.2f}".format(self.model.norm_eta))
        return self.model.pattern

    def close(self):
        if self.viewer is not None:
            # self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-m", type=int, default=9, choices=[3, 9], help="MxM beam shape")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    env = LaserControllerEnv(M=args.m)

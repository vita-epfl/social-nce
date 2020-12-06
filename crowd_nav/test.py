import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.frames import FrameStack, latest_frame
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.15)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    policy_config_file = args.policy_config

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info(' =========== TEST %s ============ ', args.policy)
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_file is None:
            logging.warning('Trainable policy is NOT specified with a model weights directory')
        else:
            try:
                policy.get_model().load_state_dict(torch.load(args.model_file))
                logging.info('Loaded policy from %s', args.model_file)
            except Exception as e:
                logging.warning(e)
                policy.get_model().load_state_dict(torch.load(args.model_file), strict=False)
                logging.info('Loaded policy partially from %s', args.model_file)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # multi-frame env
    if args.num_frames > 1:
        logging.info("stack %d frames", args.num_frames)
        env = FrameStack(env, args.num_frames)

    gamma = policy_config.getfloat('rl', 'gamma')
    explorer = Explorer(env, robot, device, args.num_frames, gamma=gamma)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            # robot.policy.safety_space = 0
            robot.policy.safety_space = args.safety_space
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = args.safety_space
        logging.warning('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            # retrieve frame stack
            if not 'RNN' in robot.policy.name:
                ob = latest_frame(ob, args.num_frames)
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos

        output_file = os.path.join(args.result_dir, 'traj_' + str(args.test_case))
        print("output_file", output_file)
        if args.traj:
            env.render('traj', output_file + '.png')
        else:
            env.render('video', output_file + '.mp4')

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.get_global_time(), info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=False)

if __name__ == '__main__':
    main()

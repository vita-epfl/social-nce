import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.frames import FrameStack
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.pretrain import freeze_model
from crowd_nav.policy.policy_factory import policy_factory

torch.manual_seed(2020)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='sail')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--imitate_config', type=str, default='configs/demonstrate.config')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='data/output/temp')
    parser.add_argument('--memory_dir', type=str, default='data/memory/temp')
    parser.add_argument('--expert_file', type=str, default='data/expert/rl_model.pth')
    parser.add_argument('--freeze', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.imitate_config = os.path.join(args.output_dir, os.path.basename(args.imitate_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.imitate_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info(' =========== Demonstrate %s ============ ', args.policy)
    logging.info('Current git head hash code: %s', (repo.head.object.hexsha))
    logging.info('Using device: %s', device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # multi-frame env
    if args.num_frames > 1:
        logging.info("stack %d frames", args.num_frames)
        env = FrameStack(env, args.num_frames)

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    # read training parameters
    if args.imitate_config is None:
        parser.error('Train config has to be specified for a trainable network')
    imitate_config = configparser.RawConfigParser()
    imitate_config.read(args.imitate_config)

    # configure demonstration and explorer
    memory = ReplayMemory(10000000)         # sufficiently large to store expert demonstration

    if not os.path.exists(args.memory_dir):
        os.makedirs(args.memory_dir)
    if robot.visible: 
        demonstration_file = os.path.join(args.memory_dir, 'data_imit_visible.pt')
    else:
        demonstration_file = os.path.join(args.memory_dir, 'data_imit_invisible.pt')

    il_episodes = imitate_config.getint('demonstration', 'il_episodes')
    expert_policy = imitate_config.get('demonstration', 'il_policy')
    expert_policy = policy_factory[expert_policy]()
    expert_policy.configure(policy_config)
    expert_policy.set_device(device)
    expert_policy.set_env(env)
    expert_policy.multiagent_training = policy.multiagent_training

    counter = imitate_config.getint('demonstration', 'counter_start')
    env.set_counter(counter)

    if robot.visible:
        expert_policy.safety_space = imitate_config.getfloat('demonstration', 'safety_space')
    else:
        expert_policy.safety_space = imitate_config.getfloat('demonstration', 'safety_space')
    robot.set_policy(expert_policy)

    noise_explore = imitate_config.getfloat('demonstration', 'noise_explore')

    gamma = policy_config.getfloat('rl', 'gamma')
    explorer = Explorer(env, robot, device, args.num_frames, memory, gamma=gamma, target_policy=policy)

    if expert_policy.trainable:
        if args.expert_file is None:
            logging.warning('Trainable policy is NOT specified with a model weights directory')
        else:
            try:
                expert_policy.get_model().load_state_dict(torch.load(args.expert_file))
                logging.info('Loaded policy from %s', args.expert_file)
            except Exception as e:
                logging.warning(e)
                expert_policy.get_model().load_state_dict(torch.load(args.expert_file), strict=False)
                logging.info('Loaded policy partially from %s', args.expert_file)
        robot.policy.set_epsilon(-1.0)

    # data collection
    robot.policy.multiagent_training = True
    explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True, noise_explore=noise_explore)

    torch.save(memory.memory, demonstration_file)
    logging.info('Save memory to %s', demonstration_file)

if __name__ == '__main__':
    main()

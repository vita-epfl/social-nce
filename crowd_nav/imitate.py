import logging
import argparse
import configparser
import os
import torch
import torch.nn as nn
import torch.optim as optim

from crowd_nav.utils.pretrain import freeze_model, trim_model_dict
from crowd_nav.utils.dataset import ImitDataset, split_dataset
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.configure import config_log, config_path
from crowd_nav.snce.contrastive import SocialNCE
from crowd_nav.snce.model import ProjHead, SpatialEncoder, EventEncoder

torch.manual_seed(2020)


def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='sail')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--contrast_sampling', type=str, default='social')
    parser.add_argument('--contrast_weight', type=float, default=1.0)
    parser.add_argument('--contrast_horizon', type=int, default=4)
    parser.add_argument('--contrast_temperature', type=float, default=0.2)
    parser.add_argument('--contrast_range', type=float, default=2.0)
    parser.add_argument('--contrast_nboundary', type=int, default=0)
    parser.add_argument('--ratio_boundary', type=float, default=0.5)
    parser.add_argument('--percent_label', type=float, default=0.5)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--model_file', type=str, default="")
    parser.add_argument('--output_dir', type=str, default='data/output/imitate')
    parser.add_argument('--memory_dir', type=str, default='data/demonstration')
    parser.add_argument('--freeze', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    args = parser.parse_args()
    return args


def build_policy(args):
    """
    Build navigation policy
    """
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        raise Exception('Policy has to be trainable')
    if args.policy_config is None:
        raise Exception('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    return policy


def set_loader(args, device):
    """
    Set Data Loader
    """
    demo_file = os.path.join(args.memory_dir, 'data_imit.pt')
    logging.info('Load data from %s', demo_file)
    data_imit = torch.load(demo_file)

    dataset_imit = ImitDataset(data_imit, None, device, horizon=args.contrast_horizon)

    validation_split = 0.3
    train_loader, valid_loader = split_dataset(dataset_imit, args.batch_size, args.percent_label, validation_split)
    return train_loader, valid_loader


def set_model(args, device):
    """
    Set policy network
    """
    policy = build_policy(args)
    policy.set_device(device)
    policy_net = policy.get_model().to(device)

    return policy_net


def load_model(policy_net, args, device):
    """
    Load pretrained model
    """
    pretrain = torch.load(args.model_file, map_location=device)
    if 'human_encoder.0.weight' in pretrain.keys():
        info = policy_net.load_state_dict(pretrain, strict=True)
    else:
        trim_state = trim_model_dict(pretrain, 'encoder')
        info = policy_net.human_encoder.load_state_dict(trim_state, strict=True)
    print(info)
    logging.info('Load pretrained model from %s', args.model_file)

    if args.freeze:
        freeze_model(policy_net, ['human_encoder'])


def train(policy_net, projection_head, encoder_sample, train_loader, criterion, nce, optimizer, args):
    """
    Jointly train the policy net and contrastive encoders
    """
    policy_net.train()
    projection_head.train()
    encoder_sample.train()
    loss_sum_all, loss_sum_task, loss_sum_nce = 0, 0, 0

    for robot_states, human_states, action_targets, pos_seeds, neg_seeds in train_loader:

        # main task
        outputs, features = policy_net(robot_states, human_states)
        loss_task = criterion(outputs, action_targets)

        # contrastive task
        if args.contrast_weight > 0:
            loss_nce = nce.loss(robot_states, human_states, pos_seeds, neg_seeds, features)
            loss = loss_task + loss_nce * args.contrast_weight
            loss_sum_nce += loss_nce.item()
        else:
            loss = loss_task

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum_all += loss.data.item()
        loss_sum_task += loss_task.item()

    num_batch = len(train_loader)
    return loss_sum_all / num_batch, loss_sum_task / num_batch, loss_sum_nce / num_batch


def validate(policy_net, projection_head, encoder_sample, valid_loader, criterion, nce, args):
    """
    Evaluate policy net
    """
    policy_net.eval()
    projection_head.eval()
    encoder_sample.eval()
    loss_sum_all, loss_sum_task, loss_sum_nce = 0, 0, 0

    with torch.no_grad():
        for robot_states, human_states, action_targets, pos_seeds, neg_seeds in valid_loader:
            outputs, features = policy_net(robot_states, human_states)
            loss_task = criterion(outputs, action_targets)
            if args.contrast_weight > 0:
                loss_nce = nce.loss(robot_states, human_states, pos_seeds, neg_seeds, features)
                loss = loss_task + loss_nce * args.contrast_weight
                loss_sum_nce += loss_nce.item()
            else:
                loss = loss_task
            loss_sum_all += loss.data.item()
            loss_sum_task += loss_task.item()

    num_batch = len(valid_loader)
    return loss_sum_all / num_batch, loss_sum_task / num_batch, loss_sum_nce / num_batch


def main():
    args = parse_arguments()
    print(args)

    # config
    if args.contrast_weight > 0:
        suffix = "-{}-data-{:.2f}-weight-{:.1f}-horizon-{:d}-temperature-{:.2f}-nboundary-{:d}".format(
            args.contrast_sampling, args.percent_label, args.contrast_weight, args.contrast_horizon, args.contrast_temperature, args.contrast_nboundary)
        if args.contrast_nboundary > 0:
            suffix += "-ratio-{:.2f}".format(args.ratio_boundary)
        if args.contrast_sampling == 'local':
            suffix += "-range-{:.2f}".format(args.contrast_range)
    else:
        suffix = "-vanilla-data-{:.2f}".format(args.percent_label)
    config_path(args, suffix)
    config_log(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # dataset
    train_loader, valid_loader = set_loader(args, device)

    # model
    policy_net = set_model(args, device)
    projection_head = ProjHead(feat_dim=64, hidden_dim=16, head_dim=8).to(device)
    if args.contrast_sampling == 'event':
        encoder_sample = EventEncoder(hidden_dim=8, head_dim=8).to(device)
    else:
        encoder_sample = SpatialEncoder(hidden_dim=8, head_dim=8).to(device)

    # pretrain
    if os.path.exists(args.model_file):
        load_model(policy_net, args, device)

    # optimize
    param = list(policy_net.parameters()) + list(projection_head.parameters()) + list(encoder_sample.parameters())
    optimizer = optim.Adam(param, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.scheduler_patience, threshold=0.01,
                                                     factor=0.5, cooldown=args.scheduler_patience, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()
    nce = SocialNCE(projection_head, encoder_sample, args.contrast_sampling, args.contrast_horizon,
                    args.contrast_nboundary, args.contrast_temperature, args.contrast_range, args.ratio_boundary)

    # loop
    for epoch in range(args.num_epoch):

        train_loss_all, train_loss_task, train_loss_nce = train(policy_net, projection_head, encoder_sample, train_loader, criterion, nce, optimizer, args)
        eval_loss_all, eval_loss_task, eval_loss_nce = validate(policy_net, projection_head, encoder_sample, valid_loader, criterion, nce, args)

        scheduler.step(train_loss_all)      # (optional) learning rate decay once training stagnates

        if epoch % args.save_every == (args.save_every - 1):
            logging.info("Epoch #%02d: loss = (%.4f, %.4f), task = (%.4f, %.4f), nce = (%.4f, %.4f)", epoch,
                         train_loss_all, eval_loss_all, train_loss_task, eval_loss_task, train_loss_nce, eval_loss_nce)
            torch.save(policy_net.state_dict(), os.path.join(args.output_dir, 'policy_net_{:02d}.pth'.format(epoch)))

    torch.save(policy_net.state_dict(), os.path.join(args.output_dir, 'policy_net.pth'))


if __name__ == '__main__':
    main()

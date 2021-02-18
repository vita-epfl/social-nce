import logging
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from crowd_nav.utils.transform import MultiAgentTransform


class ImitDataset(Dataset):

    def __init__(self, data, action_space, device, vmin=0.0, horizon=3):

        robot_data = data[0]
        human_data = data[1]
        action_data = data[2]
        value_data = data[3]

        # contrastive seeds, i.e., true positions of the robot (pos) and human neighbors (neg)
        pos_seq = torch.zeros(robot_data.size(0), horizon, 2)
        neg_seq = torch.zeros(human_data.size(0), horizon, human_data.size(1), 2)
        # remove samples at the end of episodes
        vx = robot_data[1:, 2]
        dx = robot_data[1:, 0] - robot_data[:-1, 0]
        diff = dx - vx * 0.25
        idx_done = (diff.abs() > 1e-6).nonzero(as_tuple=False)
        for t in range(horizon):
            dt = t + 1
            pos_seq[:-dt, t] = robot_data[dt:, :2]
            neg_seq[:-dt, t] = human_data[dt:, :, :2]
            for i in range(dt):
                pos_seq[idx_done-i, t] *= 0.0
                neg_seq[idx_done-i, t] *= 0.0

        # remove bad experience for imitation
        mask = (value_data > vmin).squeeze()

        self.robot_state = robot_data[mask].to(device)
        self.human_state = human_data[mask].to(device)
        self.action_target = action_data[mask].to(device)
        self.pos_state = pos_seq[mask].to(device)
        self.neg_state = neg_seq[mask].to(device)

    def __len__(self):
        return self.robot_state.size(0)

    def __getitem__(self, idx):
        return self.robot_state[idx], self.human_state[idx], self.action_target[idx], self.pos_state[idx], self.neg_state[idx]


class TrajDataset(Dataset):

    def __init__(self, data, length_pred, skip_pred, device):

        assert length_pred >= 1                 # TODO: multiple

        num_human = data[0].shape[1]
        state_dim = data[0].shape[2]

        self.transform = MultiAgentTransform(num_human)

        obsv = []
        target = []
        index = []

        for i, episode in enumerate(data):

            # remove starting and ending frame due to unpredictability
            speed = episode[:, :, -2:].norm(dim=2)
            valid = episode[(speed > 1e-4).all(axis=1)]

            length_valid = valid.shape[0]

            human_state = self.transform.transform_frame(valid)[:length_valid-length_pred*skip_pred]

            if length_valid > length_pred*skip_pred:
                upcome = []
                for k in range(length_pred):
                    propagate = episode[(k+1)*skip_pred:length_valid-(length_pred-k-1)*skip_pred, :, :2]
                    upcome.append(propagate)
                upcome = torch.cat(upcome, axis=2)
                obsv.append(human_state.view((length_valid-length_pred*skip_pred)*num_human, -1))
                target.append(upcome.view((length_valid-length_pred*skip_pred)*num_human, -1))
                index.append(torch.arange(5).repeat(length_valid-length_pred*skip_pred)+num_human*i)

        self.obsv = torch.cat(obsv).to(device)
        self.target = torch.cat(target).to(device)
        self.index = torch.cat(index).to(device)

    def __len__(self):
        return self.obsv.shape[0]

    def __getitem__(self, idx):
        return self.obsv[idx], self.target[idx]


def split_dataset(dataset, batch_size, percent_label=1.0, validation_split=0.3, is_random=False):

    dataset_size = len(dataset)
    split = int(validation_split * dataset_size)

    if is_random:
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.arange(dataset_size)

    train_indices, val_indices = indices[:int((dataset_size-split)*percent_label)], indices[-split:]

    logging.info("train_indices: %d - %d", train_indices[0].item(), train_indices[-1].item())
    logging.info("val_indices: %d - %d", val_indices[0].item(), val_indices[-1].item())

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

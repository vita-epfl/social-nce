import os
import logging
import time
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

class Trainer(object):
    def __init__(self, model, memory, device, batch_size, output_dir=''):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.output_dir = output_dir

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=20)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, actions, values = data

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        loss_meter = 0
        for _ in range(num_batches):
            inputs, actions, values = next(iter(self.data_loader))

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            loss_meter += loss.data.item()

        average_loss = loss_meter / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss

class Imitator(Trainer):

    def optimize_policy(self, num_epochs, frac_val=0.3):

        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')

        len_memory = len(self.memory)
        num_val = int(frac_val * len_memory)

        # val_memory = copy.deepcopy(self.memory)
        # del val_memory.memory[num_val:]
        # del self.memory.memory[:num_val]

        # logging.info('Memory size: train = %d, val = %d', len(self.memory.memory), len(val_memory.memory))

        criterion = nn.MSELoss(reduction='none').to(self.device)

        # train_data = DataLoader(self.memory, self.batch_size, shuffle=True)
        # val_data = DataLoader(val_memory, self.batch_size, shuffle=True)

        # indices = torch.randperm(len_memory)
        indices = torch.arange(len_memory)
        train_indices, val_indices = indices[:-num_val], indices[-num_val:]

        train_sampler = SequentialSampler(train_indices)
        valid_sampler = SequentialSampler(val_indices)

        # TODO: REMOVE
        train_data = DataLoader(self.memory, self.batch_size, sampler=train_sampler)
        val_data = DataLoader(self.memory, self.batch_size, sampler=valid_sampler)

        num_train = len(train_data)
        num_valid = len(val_data)

        timestamp = time.time()
        for epoch in range(num_epochs):

            # training
            train_loss = 0
            self.model.train()
            # print_model(self.model, ['human_encoder', 'planner'])
            # import pdb; pdb.set_trace()
            for data in train_data:
                inputs, actions, values = data
                # import pdb; pdb.set_trace()
                # print("inputs", inputs[1][0])
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, actions)
                loss = (loss * (values > 0.1)).mean()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_mse = train_loss / num_train

            # validation
            val_loss = 0
            self.model.eval()
            for data in val_data:
                inputs, actions, values = data
                outputs = self.model(inputs)
                loss = criterion(outputs, actions)
                loss = (loss * (values > 0.1)).mean()
                val_loss += loss.item()
            val_mse = val_loss / num_valid
            
            self.scheduler.step(train_loss)
            # self.scheduler.step(val_loss)

            if epoch % 10 == 0:
                logging.info('Epoch %d (%.1fs): %.4f, %.4f', epoch, time.time()-timestamp, train_mse, val_mse)
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'il_model-{:02d}.pth'.format(epoch)))
                timestamp = time.time()
        logging.info('Epoch %d (%.1fs): %.4f, %.4f', num_epochs, time.time()-timestamp, train_mse, val_mse)

    def optimize_value(self, num_epochs, frac_val=0.3):

        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')

        len_memory = len(self.memory)
        num_val = int(frac_val * len_memory)

        val_memory = copy.deepcopy(self.memory)
        del val_memory.memory[num_val:]
        del self.memory.memory[:num_val]

        logging.info('Memory size: train = %d, val = %d', len(self.memory.memory), len(val_memory.memory))

        # train_data = DataLoader(self.memory, self.batch_size, shuffle=True)
        # val_data = DataLoader(val_memory, self.batch_size, shuffle=True)

        # TODO: REMOVE 
        train_data = DataLoader(self.memory, self.batch_size, shuffle=False)
        val_data = DataLoader(val_memory, self.batch_size, shuffle=False)

        timestamp = time.time()
        for epoch in range(num_epochs):

            # training
            train_loss = 0
            self.model.train()
            for data in train_data:
                inputs, actions, values = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_mse = train_loss / math.ceil(len(self.memory.memory) / self.batch_size)

            # validation
            val_loss = 0
            self.model.eval()
            for data in val_data:
                inputs, actions, values = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                val_loss += loss.item()
            val_mse = val_loss / math.ceil(len(val_memory.memory) / self.batch_size)
            
            self.scheduler.step(train_loss)
            # self.scheduler.step(val_loss)

            if epoch % 5 == 0:
                logging.info('Epoch %d (%.1fs): %.4f, %.4f', epoch, time.time()-timestamp, train_mse, val_mse)
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'il_model-{:02d}.pth'.format(epoch)))
                timestamp = time.time()
        logging.info('Epoch %d (%.1fs): %.4f, %.4f', num_epochs, time.time()-timestamp, train_mse, val_mse)

# def print_model(model, keys):
#     for name, p in model.named_parameters():
#         if any(key in name for key in keys):
#             print("frozen parameter: ", name, p)

class SocialTrainer(object):
    def __init__(self, num_epoch, lr, output_dir):
        # settings
        self.num_epoch = num_epoch
        self.lr = lr
        self.output_dir = output_dir
        self.criterion = nn.MSELoss()

    def joint_train(self, net_imit, train_dl_imit, valid_dl_imit, net_pred, train_dl_pred, valid_dl_pred, coef=1.0):

        num_train = len(train_dl_imit)

        print("num_train: imit = {:d}, pred = {:d}".format(num_train, len(train_dl_pred)))
        assert len(train_dl_pred) >= num_train
        
        optimizer = optim.Adam(list(net_imit.parameters())+list(net_pred.parameters()), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, threshold=0.01, cooldown=50, min_lr=1e-5, verbose=True)

        timestamp = time.time()

        for epoch in range(self.num_epoch):

            net_imit.train(), net_pred.train()
            train_loss_joint, train_loss_imit, train_loss_pred = 0, 0, 0
            for data_imit, data_pred in zip(train_dl_imit, train_dl_pred):
                robot_states, human_obsv, action_targets = data_imit
                action_outputs = net_imit(robot_states, human_obsv)
                loss_imit = self.criterion(action_outputs, action_targets)

                human_states, traj_targets = data_pred
                traj_outputs = net_pred(human_states)
                loss_pred = self.criterion(traj_outputs, traj_targets)

                loss_joint = loss_imit + loss_pred * coef

                optimizer.zero_grad()
                loss_joint.backward()
                optimizer.step()

                train_loss_joint += loss_joint
                train_loss_imit += loss_imit
                train_loss_pred += loss_pred

            train_mse_joint = train_loss_joint / num_train
            train_mse_imit = train_loss_imit / num_train
            train_mse_pred = train_loss_pred / num_train

            valid_mse_imit = self.eval(net_imit, valid_dl_imit)
            valid_mse_pred = self.eval(net_pred, valid_dl_pred)

            scheduler.step(valid_mse_imit)

            if epoch % 10 == 0:
                logging.info('Epoch %d (%.1fs): %.4f, %.4f, %.4f, %.4f, %.4f', epoch, time.time()-timestamp, train_mse_joint, train_mse_imit, valid_mse_imit, train_mse_pred, valid_mse_pred)
                timestamp = time.time()

            if epoch % 100 == 0:                
                torch.save(net_imit.state_dict(), os.path.join(self.output_dir, 'il_model-{:02d}.pth'.format(epoch)))
                
        logging.info('Epoch %d (%.1fs): %.4f, %.4f, %.4f, %.4f, %.4f', self.num_epoch, time.time()-timestamp, train_mse_joint, train_mse_imit, valid_mse_imit, train_mse_pred, valid_mse_pred)
        torch.save(net_imit.state_dict(), os.path.join(self.output_dir, 'il_model.pth'))

    def train(self, model, train_loader, valid_loader):

        num_train = len(train_loader)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, threshold=0.01, cooldown=50, min_lr=1e-5, verbose=True)

        timestamp = time.time()
        for epoch in range(self.num_epoch):

            model.train()
            train_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                if len(data) > 2:
                    robot_states, human_states, targets = data
                    outputs = model(robot_states, human_states)
                else:
                    human_states, targets = data
                    outputs = model(human_states)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_mse = train_loss / num_train

            valid_mse = self.eval(model, valid_loader)

            # scheduler.step(train_mse)
            # scheduler.step(valid_mse)

            if epoch % 1 == 0:
                logging.info('Epoch %d (%.1fs): %.4f, %.4f', epoch, time.time()-timestamp, train_mse, valid_mse)
                timestamp = time.time()

            if epoch % 100 == 0:
                torch.save(model.state_dict(), os.path.join(self.output_dir, 'il_model-{:02d}.pth'.format(epoch)))

        logging.info('Epoch %d (%.1fs): %.4f, %.4f', self.num_epoch, time.time()-timestamp, train_mse, valid_mse)
        # torch.save(model.state_dict(), os.path.join(self.output_dir, 'il_model.pth'))

    def eval(self, model, valid_loader):
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data in valid_loader:
                if len(data) > 2:
                    robot_states, human_states, targets = data
                    outputs = model(robot_states, human_states)
                else:
                    human_states, targets = data
                    outputs = model(human_states)
                loss = self.criterion(outputs, targets)
                valid_loss += loss.item()
            valid_mse = valid_loss / len(valid_loader)
        return valid_mse

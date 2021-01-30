import torch
import torch.nn as nn
from crowd_nav.snce.sampling import EventSampler
from crowd_nav.utils.visualize import plot_samples

class SocialNCE():
    '''
    Social contrastive loss, encourage the extracted motion representation to be aware of socially unacceptable events
    '''

    def __init__(self, head_projection=None, encoder_sample=None, sampling='social', horizon=3, num_boundary=0, temperature=0.07, max_range=2.0, ratio_boundary=0.5):
        # encoders
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample
        # nce
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        # sampling
        self.sampling = sampling
        self.horizon = horizon
        self.device = next(head_projection.head.parameters()).device
        self.sampler = EventSampler(num_boundary, max_range, ratio_boundary, device=self.device)

    def loss(self, robot, human, pos, neg, feat):
        '''
        Input:
            robot: a tensor of shape (B, 6) for robot state, i.e. [x, y, vx, vy, gx, gy]
            human: a tensor of shape (B, N, 4) for human state, i.e. [x, y, vx, vy], where N is the number of human neighbors
            pos: a tensor of shape (B, H, 2) for postive location seeds, i.e. [x, y], where H is the sampling horizion
            neg: a tensor of shape (B, H, N, 2) for negative location seeds
            feat: a tensor of shape (B, D) for extracted features, where D is the dimension of extracted motion features
        Output:
            social nce loss
        '''

        # sampling
        if self.sampling == 'social':
            sample_pos, sample_neg, mask_valid = self.sampler.social_sampling(robot, pos[:, self.horizon-1], neg[:, self.horizon-1])
        elif self.sampling == 'local':
            sample_pos, sample_neg, mask_valid = self.sampler.local_sampling(robot, pos[:, self.horizon-1], neg[:, self.horizon-1])
        elif self.sampling == 'event':
            sample_pos, sample_neg, mask_valid = self.sampler.event_sampling(robot, pos, neg)
        elif self.sampling == 'positive':
            sample_pos, sample_neg, mask_valid, mask_pos = self.sampler.positive_sampling(robot, pos[:, self.horizon-1], neg[:, self.horizon-1])
        else:
            raise NotImplementedError

        # sanity check
        # self._sanity_check(robot, human, sample_pos, sample_neg)

        # obsv embedding, a tensor of shape (B, E) for motion embeddings
        emb_obsv = self.head_projection(feat[mask_valid])
        query = nn.functional.normalize(emb_obsv, dim=1)

        if self.sampling == 'event':
            # event embedding
            time_pos = (torch.ones(sample_pos.size(0))[:, None] * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :]).to(self.device)
            time_neg = (torch.ones(sample_neg.size(0), sample_neg.size(2))[:, None, :] * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :, None]).to(self.device)
            emb_pos = self.encoder_sample(sample_pos[mask_valid], time_pos[mask_valid, :, None])
            emb_neg = self.encoder_sample(sample_neg[mask_valid], time_neg[mask_valid, :, :, None])
            # normalized embedding
            key_pos = nn.functional.normalize(emb_pos, dim=2)
            key_neg = nn.functional.normalize(emb_neg, dim=3)
            # similarity
            sim_pos = (query[:, None, :] * key_pos).sum(dim=2)
            sim_neg = (query[:, None, None, :] * key_neg).sum(dim=3)
            # logits
            logits = torch.cat([sim_pos.view(-1).unsqueeze(1), sim_neg.view(sim_neg.size(0), -1).repeat_interleave(self.horizon, dim=0)], dim=1) / self.temperature
        else:
            # sample embedding
            emb_pos = self.encoder_sample(sample_pos[mask_valid])
            emb_neg = self.encoder_sample(sample_neg[mask_valid])
            # normalized embedding
            key_pos = nn.functional.normalize(emb_pos, dim=1)
            key_neg = nn.functional.normalize(emb_neg, dim=2)
            # pairing
            if self.sampling == 'positive':
                sim_pos = (query[:, None, :] * key_pos).sum(dim=2)
                sim_neg = (query[:, None, :] * key_neg).sum(dim=2)
                # similarity
                flat_pos = sim_pos.view(-1).unsqueeze(1)
                flat_neg = sim_neg.repeat_interleave(sim_pos.size(1), dim=0)
                # logits
                logits = (torch.cat([flat_pos, flat_neg], dim=1) / self.temperature)[mask_pos[mask_valid].view(-1)]
            else:
                # similarity
                sim_pos = (query * key_pos).sum(dim=1)
                sim_neg = (query[:, None, :] * key_neg).sum(dim=2)
                # logits
                logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature

        # loss
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        loss = self.criterion(logits, labels)

        return loss

    def _sanity_check(self, robot, human, pos, neg):
        '''
        Debug sampling strategy
        '''
        for i in range(5):
            if len(pos.shape) > 2:
                for k in range(self.horizon):
                    plot_samples(robot[i, :4], human[i], robot[i, 4:], pos[i, k], neg[i, k], fname='samples_{:d}_time_{:d}.png'.format(i, k))
            else:
                plot_samples(robot[i, :4], human[i], robot[i, 4:], pos[i], neg[i], fname='samples_{:d}.png'.format(i))

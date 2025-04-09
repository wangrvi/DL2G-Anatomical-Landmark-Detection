"MoCo builder CoordinateRestrain"
import torch
import torch.nn as nn
import numpy as np


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, TrainRestrainRadius=None, gpu=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.gpu = gpu
        self.TrainRestrainRadius = TrainRestrainRadius

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_Coordinates", torch.tensor(np.random.uniform(high=255.0, size=(3, K)).astype(np.float32)))
        # self.queue_Coordinates = torch.randint(low=0, high=3, size=(3, K), dtype=torch.int16)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, Coordinates):
        # keys.shape : N x aug_num x C
        batch_size, aug_num, C  = keys.shape
        
        # random select a aug_feature in aug_num
        random_indices = torch.randint(0, aug_num, size=(batch_size, 1))
        random_indices = torch.repeat_interleave(random_indices, C, dim=1).unsqueeze(1).cuda(self.gpu)
        selected_rows = torch.gather(keys, 1, random_indices)
        randomSelectKeys = selected_rows.squeeze(1)

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity


        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = randomSelectKeys.T
        self.queue_Coordinates[:, ptr : ptr + batch_size] = Coordinates.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    def forward(self, crop_q, crop_aug, crop_Coordinates, target):
        """
        Input:
            crop_q: a batch of query crops
            crop_aug: a batch of aug crops
            crop_Coordinates: a batch of aug crops Coordinates N*3
        Output:
            logits, targets
        """
        
        # compute query features  N=batchsize
        q = self.encoder_q(crop_q)  # queries: N xC
        q = nn.functional.normalize(q, dim=1)
        aug_num = crop_aug.shape[1]
        crop_aug = crop_aug.reshape(-1, 1, crop_aug.shape[-1], crop_aug.shape[-1], crop_aug.shape[-1])
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(crop_aug)  # keys: (N x aug_num) x C
            k = nn.functional.normalize(k, dim=1)
        k = k.reshape(q.shape[0], aug_num, -1)  # N x aug_num x C

        # compute logits
        # positive logits: Nxaug_num
        l_pos = torch.sum(q.unsqueeze(1) * k, dim=2)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        batch_SpaceRestrineWeights = self.get_batch_weights(crop_Coordinates)
        batch_SpaceRestrineWeights[l_neg <= 0] = 1
        l_neg = l_neg * batch_SpaceRestrineWeights


        # logits: Nx(aug_num+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, crop_Coordinates)

        return logits, target
    
    def get_batch_weights(self, Coordinates):
        '''First, understand the shape of these data: Coordinates N * 3, queue_label is 3 * N
            After transposing, get the required labels from queue_label
            Then find the corresponding indices and apply suppression to the initialized batch_weights by multiplying the weights at the corresponding positions
        '''
        batchsize = Coordinates.shape[0]
        # Get the suppression label range based on the label, since there are three dimensions to calculate the range, the data length is batchsize*3
        batch_Coordinates = Coordinates.reshape(-1) # (batchsize*3)
        # Need to expand it K (length of negative sample queue) times, to get unique weights for each label's negative sample queue
        boradcast_Batch_Coordinates = batch_Coordinates.repeat(self.K, 1).T  # （3*batchsize）* K
        # The shape of negative sample queue's label is 3*K, also needs to be broadcast to (3*256)*K
        queue_Coordinates = self.queue_Coordinates.clone().detach() 
        boradcast_queue_Coordinates = queue_Coordinates.repeat(batchsize, 1) # （3*batchsize）* K

        # Compare the broadcast queue with the broadcast min/max range to get a True/False matrix
        minus_Coordinates = boradcast_Batch_Coordinates - boradcast_queue_Coordinates
        batch_weights_index = torch.norm(minus_Coordinates.reshape(-1, 3, self.K), dim=1) <= self.TrainRestrainRadius# batchsize * k

        batch_weights = torch.ones((batchsize, self.K), dtype=torch.float32).cuda(self.gpu)
        batch_weights[batch_weights_index] = 0
        return batch_weights
    
        
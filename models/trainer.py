import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class Trainer:
    def __init__(self, dataset, model, batch_size=32, save_every=1, lr_decay=False,
                 warmup_tokens=375e6, final_tokens=260e9):
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.tokens = 0

        self.__setup()

        self.sampler = DistributedSampler(dataset)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            worker_init_fn=seed_worker
        )

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = model.to(self.device)
        self.optimizer = self.model.get_optimizer()
        self.loss_fn = self.model.get_loss_fn()
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.save_every = save_every

    def __setup(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        if "MASTER_ADDR" not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )

        seed = 420
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self, num_epochs=10):
        try:
            for epoch in range(num_epochs):
                self.sampler.set_epoch(epoch)

                for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)

                    if self.lr_decay:
                        batch_tokens = (targets >= 0).sum()
                        dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
                        self.tokens += batch_tokens.item()
                        if self.tokens < self.warmup_tokens:
                            lr_mult = self.tokens / max(1, self.warmup_tokens)
                        else:
                            progress = (self.tokens - self.warmup_tokens) / \
                                       max(1, self.final_tokens - self.warmup_tokens)
                            progress = min(1.0, progress)
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = self.optimizer.defaults['lr'] * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr

                    self.optimizer.step()

                    if batch_idx % 10 == 0 and self.rank == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(self.data_loader)}], Loss: {loss.item():.4f}")

                if self.rank == 0 and (epoch + 1) % self.save_every == 0:
                    torch.save(self.model.module.state_dict(), f"model_epoch_{epoch+1}.pt")
                dist.barrier()
        finally:
            self.__cleanup()

    def __cleanup(self):
        dist.destroy_process_group()

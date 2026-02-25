import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


class Trainer:
    def __init__(self, dataset, model, batch_size=32, save_every=100, lr_decay=False,
                 warmup_tokens=100_000_000, final_tokens=1_500_000_000):
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
            num_workers=5,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            prefetch_factor=2
        )

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = model.to(self.device)
        self.optimizer = self.model.get_optimizer()
        self.loss_fn = self.model.get_loss_fn()
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.model = torch.compile(self.model)

        self.global_step = 0
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

    def train(self, num_epochs=1000):
        avg_acc = 0
        try:
            for epoch in range(num_epochs):
                self.sampler.set_epoch(epoch)

                pbar = None
                if self.rank == 0:
                    pbar = tqdm(total=len(self.data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

                for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)

                    if self.rank == 0 and self.global_step % 10 == 0:
                        with torch.no_grad():
                            preds = torch.argmax(outputs, dim=-1)
                            mask = (targets != 61)

                            correct_local = (preds == targets)[mask].sum().float()
                            total_local = mask.sum().float()

                            stats = torch.tensor([correct_local, total_local], device=self.device)

                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                            avg_acc = (stats[0] / stats[1]).item()

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

                    self.global_step += 1
                    self.optimizer.step()

                    if self.rank == 0 and pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{avg_acc:.4f}",
                            "lr": f"{lr:.2e}" if self.lr_decay else self.optimizer.param_groups[0]['lr']
                        })
                        pbar.update(1)

                    if self.rank == 0 and self.global_step % self.save_every == 0:
                        checkpoint = {
                            'model_state_dict': self.model.module.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'global_step': self.global_step,
                            'epoch': epoch
                        }
                        torch.save(checkpoint, f"checkpoints/checkpoint_step_{self.global_step}.pt")
                    # dist.barrier()
        finally:
            self.__cleanup()

    def __cleanup(self):
        dist.destroy_process_group()

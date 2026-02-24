import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class Trainer:
    def __init__(self, dataset, model, optimizer, loss_fn, batch_size=32, save_every=1):
        self.__setup()

        self.sampler = DistributedSampler(dataset)
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=False,
            pin_memory=True
        )

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.optimizer = optimizer
        self.loss_fn = loss_fn

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
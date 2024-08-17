from torch.optim.lr_scheduler import _LRScheduler
import torch

class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.multi_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma, last_epoch)
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Use MultiStepLR scheduler
            return self.multi_step_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super(WarmupMultiStepLR, self).step(epoch)
        else:
            self.multi_step_scheduler.step(epoch - self.warmup_epochs)
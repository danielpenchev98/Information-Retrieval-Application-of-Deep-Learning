import torch

def generate_warmup_function(warmup_steps, training_steps):
    def warmup(current_step: int):
        if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / warmup_steps)
        else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))
    
    return warmup

def create_scheduler(transformer, warmup_fn, learning_rate=0.001):
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
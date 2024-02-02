from torch.optim import Optimizer

class NoamOpt(Optimizer):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer, factor=1):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
        self.factor = factor
    
    def state_dict(self):
        dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        dict["optimizer"] = self.optimizer.state_dict()
        return dict
    
    def load_state_dict(self, state_dict):
        base_optimizer_state = state_dict.pop("optimizer")
        self.__dict__.update(state_dict)
        self.optimizer.load_state_dict(base_optimizer_state)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 
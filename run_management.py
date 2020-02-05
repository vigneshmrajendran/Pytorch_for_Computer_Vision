from collections import OrderedDict
from collections import namedtuple
from itertools import product

import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from contextlib import contextmanager

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        #each of the variables can be encapsulated further into a class for just "Epoch"
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        #same for "run"
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(network, images)
    
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
    
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        for name, weight in self.network.named_parameters():
            self.tb.add_histogram(name, weight, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', weight.grad, self.epoch_count)
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
    def track_loss(self, loss):
        self.epoch_loss = loss.item() * self.loader.batch_size
    
    def track_num_correct(self, predictions, labels):
        self.epoch_num_correct = self._get_num_correct(predictions, labels)
    
    @torch.no_grad()
    def _get_num_correct(self, predictions, labels):
        return torch.argmax(predictions, dim=1).eq(labels).sum().item()
    
    def save(self, filename='saved_at_'+str(int(time.time()))):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')
    
    @contextmanager
    def run_setup(self, run, network, loader):
        self.begin_run(run, network, loader)
        try:
            yield
        finally:
            self.end_run()
    
    @contextmanager
    def epoch_setup(self):
        self.begin_epoch()
        try:
            yield
        finally:
            self.end_epoch()
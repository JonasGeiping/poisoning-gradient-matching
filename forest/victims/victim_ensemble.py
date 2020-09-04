"""Definition for multiple victims that share a single GPU (sequentially)."""

import torch
import numpy as np
from collections import defaultdict

from ..utils import set_random_seed, average_dicts
from ..consts import BENCHMARK
from .context import GPUContext
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase


class _VictimEnsemble(_VictimBase):
    """Implement model-specific code and behavior for multiple models on a single GPU.

    --> Running in sequential mode!

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        print(f'Initializing ensemble from random key {self.model_init_seed}.')

        self.models, self.definitions, self.criterions, self.optimizers, self.schedulers, self.epochs = [], [], [], [], [], []
        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            model, defs, criterion, optimizer, scheduler = self._initialize_model(model_name)
            self.models.append(model)
            self.definitions.append(defs)
            self.criterions.append(criterion)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            print(f'{model_name} initialized as model {idx}')
        self.defs = self.definitions[0]

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        multi_model_setup = (self.models, self.definitions, self.criterions, self.optimizers, self.schedulers)

        # Only partially train ensemble for poisoning if no poison is present
        if max_epoch is None:
            max_epoch = self.defs.epochs
        if poison_delta is None and self.args.stagger:
            # stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble)]
            # stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble + 2)[1:-1]]
            stagger_list = [int(epoch) for epoch in range(self.args.ensemble)]
            print(f'Staggered pretraining to {stagger_list}.')
        else:
            stagger_list = [max_epoch] * self.args.ensemble

        run_stats = list()
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            stats = defaultdict(list)
            model, defs, criterion, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            def loss_fn(model, outputs, labels):
                return criterion(outputs, labels)
            for epoch in range(stagger_list[idx]):
                self._step(kettle, poison_delta, loss_fn, epoch, stats, *single_model)
                if self.args.dryrun:
                    break
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            run_stats.append(stats)

        if poison_delta is None and self.args.stagger:
            average_stats = run_stats[-1]
        else:
            average_stats = average_dicts(run_stats)

        # Track epoch
        self.epochs = stagger_list

        return average_stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally minimize target loss during this.

        This function is limited because it assumes that defs.batch_size, defs.max_epoch, defs.epochs
        are equal for all models.
        """
        multi_model_setup = (self.models, self.criterions, self.optimizers, self.schedulers)

        run_stats = list()
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            model, criterion, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            def loss_fn(model, outputs, labels):
                normal_loss = criterion(outputs, labels)
                model.eval()
                if self.args.adversarial != 0:
                    target_loss = 1 / self.defs.batch_size * criterion(model(poison_targets), true_classes)
                else:
                    target_loss = 0
                model.train()
                return normal_loss + self.args.adversarial * target_loss

            self._step(kettle, poison_delta, loss_fn, self.epochs[idx], defaultdict(list), *single_model)
            self.epochs[idx] += 1
            if self.epochs[idx] > self.defs.epochs:
                self.epochs[idx] = 0
                print('Model reset to epoch 0.')
                model, criterion, optimizer, scheduler = self._initialize_model()
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            self.models[idx], self.criterions[idx], self.optimizers[idx], self.schedulers[idx] = model, criterion, optimizer, scheduler

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        [model.eval() for model in self.models]
        if dropout:
            [model.apply(apply_dropout) for model in self.models]

    def reset_learning_rate(self):
        """Reset scheduler objects to initial state."""
        for idx in range(self.args.ensemble):
            _, _, _, optimizer, scheduler = self._initialize_model()
            self.optimizers[idx] = optimizer
            self.schedulers[idx] = scheduler

    def gradient(self, images, labels, external_criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        grad_list, norm_list = [], []
        for model, criterion in zip(self.models, self.criterions):
            with GPUContext(self.setup, model) as model:
                loss = criterion(model(images), labels)
                if external_criterion is None:
                    loss = criterion(model(images), labels)
                else:
                    loss = external_criterion(model(images), labels)
                grad_list.append(torch.autograd.grad(loss, model.parameters(), only_inputs=True))
                grad_norm = 0
                for grad in grad_list[-1]:
                    grad_norm += grad.detach().pow(2).sum()
                norm_list.append(grad_norm.sqrt())
        return grad_list, norm_list

    def compute(self, function, *args):
        """Compute function on all models.

        Function has arguments that are possibly sequences of length args.ensemble
        """
        outputs = []
        for idx, (model, criterion, optimizer) in enumerate(zip(self.models, self.criterions, self.optimizers)):
            with GPUContext(self.setup, model) as model:
                single_arg = [arg[idx] if hasattr(arg, '__iter__') else arg for arg in args]
                outputs.append(function(model, criterion, optimizer, *single_arg))
        # collate
        avg_output = [np.mean([output[idx] for output in outputs]) for idx, _ in enumerate(outputs[0])]
        return avg_output

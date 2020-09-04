"""Optimization setups."""

from dataclasses import dataclass

BRITTLE_NETS = ['convnet', 'mobilenet', 'vgg', 'alexnet']  # handled with lower learning rate

def training_strategy(model_name, args):
    """Parse training strategy."""
    if args.optimization == 'conservative':
        defs = ConservativeStrategy(model_name, args)
    elif args.optimization == 'private':
        defs = PrivacyStrategy(model_name, args)
    elif args.optimization == 'adversarial':
        defs = AdversarialStrategy(model_name, args)
    elif args.optimization == 'basic':
        defs = BasicStrategy(model_name, args)
    elif args.optimization == 'memory-saving':
        defs = MemoryStrategy(model_name, args)
    else:
        defs = FastStrategy(model_name, args)
    return defs


@dataclass
class Strategy:
    """Default usual parameters, not intended for parsing."""

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    augmentations : bool
    privacy : dict
    validate : int

    def __init__(self, model_name, args):
        """Defaulted parameters. Apply overwrites from args."""
        if args.epochs is not None:
            self.epochs = args.epochs
        if args.noaugment:
            self.augmentations = False
        else:
            self.augmentations = args.data_aug
        if any(net in model_name.lower() for net in BRITTLE_NETS):
            self.lr *= 0.1

@dataclass
class ConservativeStrategy(Strategy):
    """Default usual parameters, defines a config object."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 40
        self.batch_size = 128
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.weight_decay = 5e-4
        self.augmentations = True
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10

        super().__init__(model_name, args)


@dataclass
class MemoryStrategy(Strategy):
    """Default usual parameters, defines a config object."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 0.05
        self.epochs = 40
        self.batch_size = 64
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.weight_decay = 5e-4
        self.augmentations = True
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10

        super().__init__(model_name, args)


@dataclass
class PrivacyStrategy(Strategy):
    """Enforce some eps-delta privacy by asking for nonzero (clip, noise) at the gradient level."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        if args.gradient_noise is None:
            noise = 0.01
        else:
            noise = args.gradient_noise
        if args.gradient_clip is None:
            clip = 1
        else:
            clip = args.gradient_clip
        self.lr = 0.1
        self.epochs = 40
        self.batch_size = 128
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.weight_decay = 5e-4
        self.augmentations = True
        self.privacy = dict(clip=clip, noise=noise)
        self.adversarial_steps = 0
        self.validate = 10
        super().__init__(model_name, args)

@dataclass
class BasicStrategy(Strategy):
    """Most simple stochastic gradient descent.

    This setup resembles the training procedure in MetaPoison.
    """

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 80
        self.batch_size = 128
        self.optimizer = 'SGD-basic'
        self.scheduler = 'none'
        self.weight_decay = 0
        self.augmentations = False
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10
        super().__init__(model_name, args)


@dataclass
class AdversarialStrategy(Strategy):
    """Implement adversarial training to defend against the poisoning."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 40
        self.batch_size = 128
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.weight_decay = 5e-4
        self.augmentations = True
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 4
        self.validate = 10
        super().__init__(model_name, args)

@dataclass
class FastStrategy(Strategy):
    """Default fast strategy.

    Possible future options for FastStrategy.
    - mixup regularization!
    - label smoothing!
    - cosine weight_decay with learning rate warmup
    - Massive batch sizes with linear lr scaling
    - no bias decay?
    """

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 0.1 * 512 / 128
        self.epochs = 40
        self.batch_size = 512
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.weight_decay = 5e-4 * 512 / 128
        self.augmentations = True
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 20
        super().__init__(model_name, args)

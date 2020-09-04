"""Implement victim behavior, for single-victim, ensemble and stuff."""
import torch

from .victim_distributed import _VictimDistributed
from .victim_ensemble import _VictimEnsemble
from .victim_single import _VictimSingle

def Victim(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.local_rank is not None:
        return _VictimDistributed(args, setup)
    elif args.ensemble == 1:
        return _VictimSingle(args, setup)
    elif args.ensemble > 1:
        return _VictimEnsemble(args, setup)


from .optimization_strategy import training_strategy
__all__ = ['Victim', 'training_strategy']

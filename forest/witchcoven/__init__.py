"""Interface for poison recipes."""

import torch

from .witch_bullseye import WitchBullsEye
from .witch_matching import WitchGradientMatching, WitchGradientMatchingNoisy
from .witch_metapoison import WitchMetaPoison
from .witch_poison_frogs import WitchFrogs
from .witch_watermark import WitchWatermark


def Witch(args, setup=dict(device=torch.device("cuda"), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == "gradient-matching":
        return WitchGradientMatching(args, setup)
    elif args.recipe == "gradient-matching-private":
        return WitchGradientMatchingNoisy(args, setup)
    elif args.recipe == "watermark":
        return WitchWatermark(args, setup)
    elif args.recipe == "metapoison":
        return WitchMetaPoison(args, setup)
    elif args.recipe == "poison-frogs":
        return WitchFrogs(args, setup)
    elif args.recipe == "bullseye":
        return WitchBullsEye(args, setup)
    else:
        raise NotImplementedError()


__all__ = ["Witch"]

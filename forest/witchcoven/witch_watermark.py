"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch




class WitchWatermark(_Witch):
    """Brew poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _brew(self, victim, kettle):
        """Sanity check: Contructing data poisons by watermarking."""
        # Compute target gradients
        self._initialize_brew(victim, kettle)
        poison_delta = kettle.initialize_poison()
        poison_imgs = torch.stack([data[0] for data in kettle.poisonset], dim=0).to(**self.setup)

        for poison_id, (img, label, image_id) in enumerate(kettle.poisonset):
            poison_img = img.to(**self.setup)

            target_id = poison_id % len(kettle.targetset)

            # Place
            delta_slice = self.targets[target_id] - poison_img
            delta_slice *= self.args.eps / 255

            # Project
            delta_slice = torch.max(torch.min(delta_slice, self.args.eps / kettle.ds / 255), -self.args.eps / kettle.ds / 255)
            delta_slice = torch.max(torch.min(delta_slice, (1 - kettle.dm) / kettle.ds - poison_img), -kettle.dm / kettle.ds - poison_img)
            poison_delta[poison_id] = delta_slice.cpu()

        return poison_delta.cpu()

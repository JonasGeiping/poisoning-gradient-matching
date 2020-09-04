"""Helper functions for context managing."""
import torch

class GPUContext():
    """GPU context for quick (code-wise) moves to and from GPU."""

    def __init__(self, setup, model):
        """Init with setup info."""
        self.setup = setup
        self.model = model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def __enter__(self):
        """Enter."""
        return self.model

    def __exit__(self, type, value, traceback):
        """Return model to CPU."""
        if torch.cuda.device_count() > 1:
            model = self.model.module
        self.model.to(device=torch.device('cpu'))

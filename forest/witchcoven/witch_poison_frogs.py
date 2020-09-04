"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchFrogs(_Witch):
    """Brew poison frogs poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, targets, intended_classes, true_classes):
        """Implement the closure here."""
        def closure(model, criterion, optimizer, target_grad, target_clean_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                pass  # use the default for untargeted or targeted cross entropy
            # Carve up the model
            feature_model, last_layer = self.bypass_last_layer(model)


            # Get standard output:
            outputs = feature_model(inputs)
            outputs_targets = feature_model(targets)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()

            feature_loss = (outputs.mean(dim=0) - outputs_targets.mean(dim=0)).pow(2).mean()
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure


    @staticmethod
    def bypass_last_layer(model):
        """Hacky way of separating features and classification head for many models.

        Patch this function if problems appear.
        """
        layer_cake = list(model.children())
        last_layer = layer_cake[-1]
        headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<
        return headless_model, last_layer

"""Main class, holding information about models and training/testing routines."""

import torch

from collections import OrderedDict
from ..utils import cw_loss

from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK
from .modules import MetaMonkey

from .witch_base import _Witch


class WitchMetaPoison(_Witch):
    """Brew metapoison with given arguments.

    Note: This function does not work in single-model-multi-GPU mode, due to the weights being fixed to a single GPU.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, targets, intended_classes, *args):
        def closure(model, criterion, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                pass  # use the default for untargeted or targeted cross entropy

            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            # model.train()
            model = MetaMonkey(model)

            for _ in range(self.args.nadapt):
                outputs = model(inputs, model.parameters)
                prediction = (outputs.data.argmax(dim=1) == labels).sum()

                poison_loss = criterion(outputs, labels)
                poison_grad = torch.autograd.grad(poison_loss, model.parameters.values(),
                                                  retain_graph=True, create_graph=True, only_inputs=True)

                current_lr = optimizer.param_groups[0]['lr']
                model.parameters = OrderedDict((name, param - current_lr * grad_part)
                                               for ((name, param), grad_part) in zip(model.parameters.items(), poison_grad))
            # model.eval()
            target_outs = model(targets, model.parameters)
            target_loss = criterion(target_outs, intended_classes)
            target_loss.backward(retain_graph=self.retain)

            return target_loss.detach().cpu(), prediction.detach().cpu()
        return closure

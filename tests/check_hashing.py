"""Check random seeding by hash."""

import unittest

import forest
import torch


class TestHashing(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Spoof args."""
        args = forest.options().parse_args()
        self.defs = forest.ConservativeStrategy(args)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float

        self.args = args
        self.setup = dict(device=device, dtype=dtype)

    def test_modelkey(self):
        self.args.modelkey = 3213894809
        model = forest.Victim(self.args, self.defs, setup=self.setup)
        model.initialize(self.args.modelkey)
        self.assertAlmostEqual(model.model.linear.weight[0][0].item(), 0.007770763710141182, places=5)

    def test_poisonkey(self):
        self.args.poisonkey = 280878748
        data = forest.Kettle(self.args, self.defs, setup=self.setup)

        class_names = data.trainset.classes
        self.assertTrue(class_names[data.targetset[0][1]] == 'cat')
        self.assertTrue(class_names[data.poisonset[0][1]] == 'bird')

if __name__ == '__main__':
    unittest.main()

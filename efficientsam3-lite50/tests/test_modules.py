"""PyTorch-only tests for modular transformations."""
import unittest
import torch
from torch import nn
from lite50.modules import LiteProjectionHead, LiteFPNLevel, FactorizedTokenEmbedding
from lite50.builder import _prune, Lite50Config

class TestModules(unittest.TestCase):
    def test_v02_defaults(self):
        config = Lite50Config()
        self.assertEqual(config.text_embedding_rank, 128)
        self.assertEqual(config.geometry_layers, 3)
        self.assertEqual(config.parameter_budget, 50_000_000)

    def test_rank_ablations(self):
        vocab, dim = 49408, 512
        for rank in (64, 96, 128):
            m = FactorizedTokenEmbedding(vocab, dim, rank)
            count = sum(p.numel() for p in m.parameters())
            self.assertEqual(count, vocab * rank + rank * dim)
        self.assertEqual(49408 * (128 - 64) + (128 - 64) * 512, 3_194_880)

    def test_projection(self):
        m = LiteProjectionHead().eval()
        with torch.no_grad():
            self.assertEqual(tuple(m(torch.randn(1,512,4,4)).shape), (1,1024,4,4))
        self.assertLess(sum(p.numel() for p in m.parameters()), 500_000)
    def test_fpn(self):
        x = torch.randn(1,1024,4,4)
        for s, side in [(4.,16),(2.,8),(1.,4),(.5,2)]:
            with torch.no_grad():
                output = LiteFPNLevel(scale=s).eval()(x)
            self.assertEqual(tuple(output.shape), (1,256,side,side))
    def test_low_rank_embedding(self):
        m = FactorizedTokenEmbedding(1024,512,32)
        m.initialize_from_dense(torch.randn(1024,512), niter=0)
        self.assertEqual(tuple(m(torch.tensor([[3,1,2]])).shape),(1,3,512))
        self.assertTrue(torch.isfinite(m(torch.tensor([[3,1,2]]))).all())
    def test_prune(self):
        a = nn.ModuleList([nn.Linear(4,4) for _ in range(6)])
        b = _prune(a,4)
        self.assertEqual(len(b),4)
        self.assertIs(a[0],b[0])
        self.assertIs(a[-1],b[-1])
if __name__ == "__main__":
    unittest.main()

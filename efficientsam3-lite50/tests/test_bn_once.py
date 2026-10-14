import copy
import unittest

import torch
from torch import nn

from lite50.bn_once import BatchNormOncePerImage


class BatchNormOnceTests(unittest.TestCase):
    def test_prompt_chunks_preserve_image_statistics_and_gradients(self):
        for momentum in (0.1, None):
            with self.subTest(momentum=momentum):
                torch.manual_seed(123)
                initial = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2, momentum=momentum)).double()
                image = torch.randn(1, 3, 7, 7, dtype=torch.float64)
                targets = torch.randn(8, 2, 7, 7, dtype=torch.float64)
                full, chunked, naive = [copy.deepcopy(initial) for _ in range(3)]
                (full(image).expand_as(targets) - targets).square().mean().backward()
                guard = BatchNormOncePerImage(chunked)
                for start, end in ((0, 3), (3, 6), (6, 8)):
                    for model in (chunked, naive):
                        prediction = model(image).expand(end-start, -1, -1, -1)
                        loss = (prediction - targets[start:end]).square().mean() * (end-start)/8
                        loss.backward()
                    guard.after_backward()
                self.assertEqual(int(naive[1].num_batches_tracked), 3)
                self.assertEqual(int(chunked[1].num_batches_tracked), 1)
                self.assertEqual(set(full.state_dict()), set(chunked.state_dict()))
                for name, value in full.named_buffers():
                    self.assertTrue(torch.equal(value, dict(chunked.named_buffers())[name]), name)
                for expected, actual in zip(full.parameters(), chunked.parameters()):
                    self.assertIsNotNone(actual.grad)
                    torch.testing.assert_close(expected.grad, actual.grad, rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(full.eval()(image), chunked.eval()(image), rtol=0, atol=0)
                # A fresh logical image must commit its own update.
                chunked.train().zero_grad(set_to_none=True)
                second = BatchNormOncePerImage(chunked)
                chunked(image + 1).square().mean().backward()
                second.after_backward()
                self.assertEqual(int(chunked[1].num_batches_tracked), 2)

    def test_eval_and_stateless_bn_are_not_modified(self):
        model = nn.Sequential(nn.BatchNorm2d(2), nn.BatchNorm2d(2, track_running_stats=False))
        model[0].eval()
        before = {k: v.clone() for k, v in model[0].state_dict().items()}
        guard = BatchNormOncePerImage(model)
        for _ in range(2):
            model(torch.randn(1, 2, 3, 3)).square().mean().backward()
            guard.after_backward()
        for name, value in model[0].state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == '__main__':
    unittest.main()

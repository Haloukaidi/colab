"""Safety gates for official-style GT-supervised Lite50 Fusion Stage3.

Offline config assertions run without the SAM3 package. Full builder/loss tests
need the pinned official checkout (set SAM3_ROOT and PYTHONPATH as in the repo).
"""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class SupervisedConfigTests(unittest.TestCase):
    def test_default_fusion_has_no_teacher_and_uses_real_masks(self):
        import yaml
        config = yaml.safe_load((ROOT / "configs/distill_stage3.yaml").read_text())
        model = config["trainer"]["model"]
        self.assertEqual(model["_target_"], "lite50.stage3.build_stage3_supervised")
        self.assertNotIn("teacher_checkpoint", model)
        self.assertNotIn("teacher_cache_dir", model)
        self.assertNotIn("mask_kd_weight", model)
        self.assertNotIn("teacher_checkpoint", config["paths"])
        self.assertEqual(
            config["trainer"]["loss"]["all"]["_target_"],
            "lite50.stage3.SupervisedLoss",
        )
        supervised = config["trainer"]["loss"]["all"]["supervised"]
        self.assertEqual(
            supervised["_target_"], "sam3.train.loss.sam3_loss.Sam3LossWrapper"
        )
        loss_types = {item["_target_"] for item in supervised["loss_fns_find"]}
        self.assertIn("sam3.train.loss.loss_fns.Masks", loss_types)
        self.assertIn("sam3.train.loss.loss_fns.Boxes", loss_types)
        self.assertIn("sam3.train.loss.loss_fns.IABCEMdetr", loss_types)
        self.assertTrue(config["trainer"]["data"]["train"]["collate_fn"]["with_seg_masks"])
        self.assertEqual(config["trainer"]["data"]["train"]["dataset"]["split"], "train")

    def test_previous_kd_config_is_preserved_as_explicit_ablation(self):
        import yaml
        kd = yaml.safe_load((ROOT / "configs/distill_stage3_kd.yaml").read_text())
        geo = yaml.safe_load((ROOT / "configs/distill_geometry.yaml").read_text())
        for cfg in (kd, geo):
            self.assertEqual(
                cfg["trainer"]["model"]["_target_"],
                "lite50.stage3.build_stage3_distiller",
            )
            self.assertEqual(
                cfg["trainer"]["loss"]["all"]["_target_"],
                "lite50.stage3.DistillationLoss",
            )
            self.assertIn("teacher_checkpoint", cfg["trainer"]["model"])
        self.assertEqual(geo["phase"], "geometry")

    def test_default_launcher_uses_supervised_recipe(self):
        code = (ROOT / "scripts/train_distill_stage3.py").read_text()
        self.assertIn('configs/distill_stage3.yaml', code)
        self.assertIn('builder == "lite50.stage3.build_stage3_supervised"', code)
        self.assertIn('"teacher_checkpoint", "teacher_cache_dir"', code)


@unittest.skipUnless(os.environ.get("SAM3_ROOT"), "Requires pinned official SAM3 training dependencies")
class SupervisedRuntimeTests(unittest.TestCase):
    class FakeStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(2.0))

        def forward(self, batch):
            return [{"pred_masks": self.value * batch}]

        def back_convert(self, target):
            return {"target": target}

    def test_build_model_never_loads_sam3_teacher(self):
        from lite50.stage3 import build_stage3_supervised
        student = self.FakeStudent()
        with (patch("lite50.stage3.build_student", return_value=student) as build_student,
              patch("lite50.stage3.build_teacher",
                    side_effect=AssertionError("Teacher must not be constructed")) as build_teacher):
            model = build_stage3_supervised("rv-m.pt", "geometry.pt", "fusion1", "cpu")
        build_student.assert_called_once()
        build_teacher.assert_not_called()
        self.assertFalse(hasattr(model, "teacher"))
        self.assertNotIn("teacher_checkpoint", model.provenance)
        self.assertEqual(model.provenance["objective"], "supervised")
        self.assertEqual(set(model.state_dict()), {"student.value"})
        tensor = torch.tensor([[[[1.]]]])
        result = model(tensor)
        self.assertTrue(torch.equal(result[0]["pred_masks"], 2 * tensor))
        self.assertEqual(model.back_convert("masks"), {"target": "masks"})

    def test_supervised_loss_uses_only_upstream_ground_truth(self):
        from lite50.stage3 import SupervisedLoss

        class FakeUpstreamLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, outputs, targets):
                self.calls += 1
                masks = next(iter(outputs))["pred_masks"]
                return {"core_loss": (masks - targets).square().mean(),
                        "loss_mask": masks.abs().mean()}

        original = FakeUpstreamLoss()
        loss_fn = SupervisedLoss(supervised=original)
        predicted = torch.tensor([[[[2.]]]], requires_grad=True)
        result = loss_fn([{"pred_masks": predicted}], torch.tensor([[[[1.]]]]))
        self.assertEqual(original.calls, 1)
        self.assertEqual(set(result), {"core_loss", "loss_mask"})
        self.assertAlmostEqual(result["core_loss"].item(), 1.0)
        result["core_loss"].backward()
        self.assertTrue(torch.equal(predicted.grad, torch.ones_like(predicted) * 2))

    def test_supervised_loss_refuses_accidental_kd_outputs(self):
        from lite50.stage3 import SupervisedLoss
        loss_fn = SupervisedLoss(supervised=nn.Identity())
        with self.assertRaisesRegex(ValueError, "KD predictions"):
            loss_fn([{"pred_masks": torch.ones(1), "lite50_kd": {}}], None)

    def test_supervised_builder_requires_previous_checkpoint_and_fusion_stage(self):
        from lite50.stage3 import build_stage3_supervised
        with self.assertRaisesRegex(ValueError, "requires a Lite50 checkpoint"):
            build_stage3_supervised("rv.pt", None, "fusion1")
        with self.assertRaisesRegex(ValueError, "fusion1/fusion2/fusion4"):
            build_stage3_supervised("rv.pt", "prior.pt", "geometry")


if __name__ == "__main__":
    unittest.main()

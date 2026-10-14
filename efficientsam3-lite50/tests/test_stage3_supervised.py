"""Compare default Lite50 SACap Stage3 with the pinned official RepViT recipe.

Offline checks read only YAML; runtime checks require installed official SAM3
and `SAM3_ROOT` (as for the existing Lite50 smoke tests).
"""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class OfficialSupervisedConfigTests(unittest.TestCase):
    def test_default_matches_official_sacap_stage3_intent(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "configs/distill_stage3.yaml").read_text())
        self.assertEqual(cfg["phase"], "stage3")
        self.assertEqual(cfg["prompt_mode"], "text")  # official use_prompt_boxes=false
        self.assertNotIn("teacher_checkpoint", cfg["paths"])
        self.assertNotIn("lr_text", cfg)
        self.assertNotIn("lr_fusion", cfg)
        self.assertEqual(cfg["trainer"]["model"]["_target_"],
                         "lite50.stage3.build_stage3_supervised")
        self.assertEqual(cfg["trainer"]["loss"]["all"]["_target_"],
                         "lite50.stage3.SupervisedLoss")
        model = cfg["trainer"]["model"]
        self.assertNotIn("teacher_checkpoint", model)
        self.assertNotIn("train_text_in_fusion", model)
        self.assertTrue(cfg["trainer"]["data"]["train"]["collate_fn"]["with_seg_masks"])
        self.assertEqual(cfg["trainer"]["data"]["train"]["dataset"]["split"], "train")
        sup = cfg["trainer"]["loss"]["all"]["supervised"]
        self.assertEqual(sup["_target_"], "sam3.train.loss.sam3_loss.Sam3LossWrapper")
        funcs = {x["_target_"]: x for x in sup["loss_fns_find"]}
        self.assertIn("sam3.train.loss.loss_fns.Masks", funcs)
        self.assertIn("sam3.train.loss.loss_fns.Boxes", funcs)
        cls = funcs["sam3.train.loss.loss_fns.IABCEMdetr"]
        self.assertFalse(cls["weak_loss"])
        self.assertFalse(cls["pos_focal"])
        self.assertEqual(cls["pad_n_queries"], 128)
        self.assertEqual(cls["pad_scale_pos"], 1.0)
        masks = funcs["sam3.train.loss.loss_fns.Masks"]["weight_dict"]
        self.assertEqual(masks["loss_mask"], 200.0)
        self.assertEqual(masks["loss_dice"], 10.0)
        sched = cfg["trainer"]["optim"]["options"]["lr"]
        pats = " ".join(" ".join(x.get("param_names", [])) for x in sched)
        self.assertIn("student.backbone.vision_backbone.trunk.model.backbone.*", pats)
        self.assertIn("student.backbone.vision_backbone.trunk.model.head.*", pats)
        self.assertIn("student.backbone.vision_backbone.convs.*", pats)
        self.assertNotIn("language_backbone", pats)
        self.assertNotIn("transformer.encoder", pats)

    def test_kd_and_lite50_fusion_remain_explicit_experiments(self):
        import yaml
        kd = yaml.safe_load((ROOT / "configs/distill_stage3_kd.yaml").read_text())
        geo = yaml.safe_load((ROOT / "configs/distill_geometry.yaml").read_text())
        fusion = yaml.safe_load(
            (ROOT / "configs/distill_stage3_fusion_experimental.yaml").read_text()
        )
        for c in (kd, geo):
            self.assertEqual(c["trainer"]["model"]["_target_"],
                             "lite50.stage3.build_stage3_distiller")
            self.assertEqual(c["trainer"]["loss"]["all"]["_target_"],
                             "lite50.stage3.DistillationLoss")
            self.assertIn("teacher_checkpoint", c["trainer"]["model"])
        self.assertEqual(geo["phase"], "geometry")
        self.assertEqual(fusion["phase"], "fusion1")
        self.assertEqual(fusion["trainer"]["model"]["_target_"],
                         "lite50.stage3.build_stage3_supervised")

    def test_launcher_enforces_kd_supervision_separation(self):
        code = (ROOT / "scripts/train_distill_stage3.py").read_text()
        self.assertIn("configs/distill_stage3.yaml", code)
        self.assertIn('builder == "lite50.stage3.build_stage3_supervised"', code)
        self.assertIn('"teacher_checkpoint", "teacher_cache_dir"', code)


@unittest.skipUnless(os.environ.get("SAM3_ROOT"),
                     "Requires pinned official SAM3 training dependencies")
class OfficialSupervisedRuntimeTests(unittest.TestCase):
    class FakeStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(2.0))

        def forward(self, batch):
            return [{"pred_masks": self.value * batch}]

        def back_convert(self, target):
            return {"target": target}

    def test_build_model_never_instantiates_teacher(self):
        from lite50.stage3 import build_stage3_supervised
        student = self.FakeStudent()
        with (patch("lite50.stage3.build_student", return_value=student) as build_student,
              patch("lite50.stage3.build_teacher",
                    side_effect=AssertionError("Unexpected SAM3 teacher")) as build_teacher):
            model = build_stage3_supervised("rv-m.pt", "geometry.pt", "stage3", "cpu")
        build_student.assert_called_once_with("rv-m.pt", "geometry.pt", "stage3", "cpu")
        build_teacher.assert_not_called()
        self.assertFalse(hasattr(model, "teacher"))
        self.assertEqual(model.provenance["objective"], "supervised")
        self.assertNotIn("teacher_checkpoint", model.provenance)
        self.assertEqual(set(model.state_dict()), {"student.value"})
        batch = torch.ones(1, 1, 1, 1)
        self.assertTrue(torch.equal(model(batch)[0]["pred_masks"], 2 * batch))
        self.assertEqual(model.back_convert("x"), {"target": "x"})

    def test_official_phase_trains_vision_trunk_projection_fpn_only(self):
        from test_distillation import dummy_model
        from lite50.train_policy import apply_train_policy, build_optimizer, assert_optimizer_coverage
        model = dummy_model()
        report = apply_train_policy(model, "stage3")
        self.assertEqual(set(report["groups"]),
                         {"repvit", "projection", "fpn0", "fpn1", "fpn2"})
        self.assertTrue(any(p.requires_grad for p in model.backbone.vision_backbone.trunk.model.backbone.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.backbone.vision_backbone.trunk.model.head.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.language_backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.transformer.encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.geometry_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.segmentation_head.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.vision_backbone.convs[3].parameters()))
        self.assertFalse(model.backbone.vision_backbone.trunk.model.backbone[1].training)
        model.eval().train()
        self.assertFalse(model.backbone.vision_backbone.trunk.model.backbone[1].training)
        assert_optimizer_coverage(model, build_optimizer(model))

    def test_supervised_loss_uses_only_official_gt_loss(self):
        from lite50.stage3 import SupervisedLoss

        class FakeGT(nn.Module):
            def forward(self, outputs, target):
                masks = next(iter(outputs))["pred_masks"]
                return {"core_loss": (masks - target).square().mean(),
                        "loss_mask": masks.abs().mean()}

        out = torch.tensor([[[[2.0]]]], requires_grad=True)
        result = SupervisedLoss(FakeGT())(
            [{"pred_masks": out}], torch.tensor([[[[1.0]]]])
        )
        self.assertEqual(set(result), {"core_loss", "loss_mask"})
        result["core_loss"].backward()
        self.assertTrue(torch.equal(out.grad, torch.full_like(out, 2.0)))

    def test_supervised_rejects_kd_and_missing_checkpoints(self):
        from lite50.stage3 import SupervisedLoss, build_stage3_supervised
        with self.assertRaisesRegex(ValueError, "KD predictions"):
            SupervisedLoss(nn.Identity())(
                [{"pred_masks": torch.ones(1), "lite50_kd": {}}], None
            )
        with self.assertRaisesRegex(ValueError, "requires a Lite50 checkpoint"):
            build_stage3_supervised("rv.pt", None, "stage3")
        with self.assertRaisesRegex(ValueError, "stage3 or fusion1"):
            build_stage3_supervised("rv.pt", "some.pt", "geometry")


if __name__ == "__main__":
    unittest.main()

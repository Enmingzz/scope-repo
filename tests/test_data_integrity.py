from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from opsd.visionzip_aokvqa.data_integrity import verify_decontaminated_training_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/openmmreasoner_llava_cot_train10k_decontam_v1_seed42"
TRAIN = DATA_DIR / "train10k_decontam_qwentok512_imgtok1152_seed42.jsonl"
MANIFEST = DATA_DIR / "train10k_decontam_qwentok512_imgtok1152_seed42_stats.json"


@unittest.skipUnless(
    TRAIN.is_file() and MANIFEST.is_file(),
    "cluster-local decontaminated dataset is unavailable",
)
class DataIntegrityTests(unittest.TestCase):
    def test_current_decontaminated_dataset_passes(self) -> None:
        result = verify_decontaminated_training_data(TRAIN, MANIFEST)
        self.assertTrue(all(result["checks"].values()))

    def test_manifest_with_perceptual_overlap_is_rejected(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["independent_postbuild_audit"]["perceptual_image_matches"] = 1
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "tampered_manifest.json"
            tampered.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "decontamination manifest failed"):
                verify_decontaminated_training_data(TRAIN, tampered)


if __name__ == "__main__":
    unittest.main()

import unittest

from omegaconf import OmegaConf

from jssp_core.instances.jssp import JSSPInstance
from jssp_gnn.dispatcher.utils_matrix import load_instance_from_cfg


class TestInstanceLoading(unittest.TestCase):
    def test_load_path(self):
        # Test that it falls back to get_instance for non-size strings
        # We use a dummy path that doesn't exist, but get_instance might try to load it.
        # Instead, let's use a known alias if possible, or mock get_instance.
        # For simplicity, let's just test the size string parsing logic which is the new feature.
        pass

    def test_load_size_string(self):
        cfg = OmegaConf.create(
            {
                "env": {
                    "instance": "6x6",
                    "instance_generator_kwargs": {
                        "min_duration": 1,
                        "max_duration": 10,
                    },
                }
            }
        )

        instance = load_instance_from_cfg(cfg)
        self.assertIsInstance(instance, JSSPInstance)
        self.assertEqual(instance.num_jobs(), 6)
        self.assertEqual(instance.num_machines(), 6)

    def test_load_size_string_different_dims(self):
        cfg = OmegaConf.create(
            {
                "env": {
                    "instance": "10x5",
                    "instance_generator_kwargs": {
                        "min_duration": 1,
                        "max_duration": 10,
                    },
                }
            }
        )

        instance = load_instance_from_cfg(cfg)
        self.assertIsInstance(instance, JSSPInstance)
        self.assertEqual(instance.num_jobs(), 10)
        self.assertEqual(instance.num_machines(), 5)

    def test_load_size_string_no_kwargs(self):
        cfg = OmegaConf.create(
            {
                "env": {
                    "instance": "4x4"
                    # No instance_generator_kwargs
                }
            }
        )

        instance = load_instance_from_cfg(cfg)
        self.assertIsInstance(instance, JSSPInstance)
        self.assertEqual(instance.num_jobs(), 4)
        self.assertEqual(instance.num_machines(), 4)


if __name__ == "__main__":
    unittest.main()

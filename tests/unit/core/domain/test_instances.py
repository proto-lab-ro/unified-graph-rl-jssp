"""
Unit tests for JSSP instance handling.
"""

import random

import pytest

from jssp_core.instances import (
    F3X3_INSTANCE,
    _load_instance,
    _parse_instance,
    generate_random_jssp_instance,
    get_instance,
    get_instance_info,
    read_yaml_specification_instance,
    read_yaml_specification_logistics,
    validate_instance,
)
from jssp_core.instances.jssp import save_instance


@pytest.mark.unit
class TestInstanceParsing:
    """Test instance parsing functionality."""

    def test_parse_ft06_instance(self, ft06_instance):
        """Test parsing of FT06 instance."""
        assert len(ft06_instance) == 6  # 6 jobs
        assert len(ft06_instance[0]) == 6  # 6 operations per job

        # Check first operation of first job
        machine_id, duration = ft06_instance[0][0]
        assert isinstance(machine_id, int)
        assert isinstance(duration, int)
        assert machine_id >= 0
        assert duration > 0

    def test_parse_3x3_instance(self, small_3x3_instance):
        """Test parsing of 3x3 instance."""
        assert len(small_3x3_instance) == 3  # 3 jobs
        assert len(small_3x3_instance[0]) == 3  # 3 operations per job

        # Verify structure
        for job in small_3x3_instance:
            assert len(job) == 3
            for machine_id, duration in job:
                assert isinstance(machine_id, int)
                assert isinstance(duration, int)
                assert 0 <= machine_id < 3  # 3 machines
                assert duration > 0

    def test_parse_empty_instance(self):
        """Test parsing of empty instance."""
        empty_text = ""
        with pytest.raises(ValueError):
            _parse_instance(empty_text)

    def test_parse_invalid_format(self):
        """Test parsing with invalid format raises appropriate error."""
        invalid_text = "not a valid instance format"
        with pytest.raises((ValueError, IndexError)):
            _parse_instance(invalid_text)


@pytest.mark.unit
class TestRandomInstanceGeneration:
    """Test random instance generation."""

    def test_generate_random_instance_basic(self):
        """Test basic random instance generation."""
        num_jobs = 3
        num_machines = 3
        instance = generate_random_jssp_instance(num_jobs, num_machines)

        assert len(instance) == num_jobs
        for job in instance:
            assert len(job) == num_machines
            for machine_id, duration in job:
                assert 0 <= machine_id < num_machines
                assert duration > 0

    def test_generate_random_instance_with_seed(self):
        """Test random instance generation with seed for reproducibility."""
        num_jobs = 2
        num_machines = 2

        random.seed(42)
        instance1 = generate_random_jssp_instance(num_jobs, num_machines)
        random.seed(42)
        instance2 = generate_random_jssp_instance(num_jobs, num_machines)

        assert instance1 == instance2

    def test_generate_random_instance_different_seeds(self):
        """Test that different seeds produce different instances."""
        num_jobs = 3
        num_machines = 3

        random.seed(1)
        instance1 = generate_random_jssp_instance(num_jobs, num_machines)
        random.seed(2)
        instance2 = generate_random_jssp_instance(num_jobs, num_machines)

        assert instance1 != instance2
        assert instance1 != instance2


@pytest.mark.unit
class TestInstanceLoading:
    """Test instance loading from files."""

    def test_load_instance_from_file(self, tmp_path):
        """Test loading instance from temporary file."""
        # Create temporary instance file
        instance_content = F3X3_INSTANCE
        temp_file = tmp_path / "test_instance.txt"
        temp_file.write_text(instance_content)

        # Load and verify
        loaded_instance = _load_instance(str(temp_file))
        expected_instance = _parse_instance(F3X3_INSTANCE)

        assert loaded_instance == expected_instance

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_instance("nonexistent_file.txt")

    def test_save_instance_roundtrip(self, tmp_path):
        """Instances saved to disk can be parsed back correctly."""
        instance = get_instance("f3x3")
        path = tmp_path / "roundtrip.txt"
        save_instance(instance, path, instance_name="test", comment="unit test")

        reloaded = _parse_instance(path.read_text())
        assert reloaded == instance


@pytest.mark.unit
class TestInstanceUtilities:
    """Test helper utilities for JSSP instances."""

    def test_get_instance_name_resolution(self, tmp_path):
        """Factory should resolve benchmark names by checking jssp_instances/."""
        # Check standard name ft06 (supported via file or fallback)
        inst = get_instance("ft06")
        assert len(inst) == 6

        # Check name that only exists as file in jssp_instances/
        inst_ft10 = get_instance("ft10")
        assert len(inst_ft10) == 10

        # Check direct path
        file_path = tmp_path / "inst.txt"
        file_path.write_text(F3X3_INSTANCE)
        from_path = get_instance(str(file_path))
        assert from_path == _parse_instance(F3X3_INSTANCE)

        # Check raw text
        raw_text = "0 5\n"
        from_text = get_instance(raw_text)
        assert len(from_text) == 1
        assert from_text[0] == [(0, 5)]

    def test_get_instance_spec_dict_random_and_uniform(self):
        """Dict specs should dispatch to generators."""
        random_inst = get_instance(
            {"type": "random", "num_jobs": 2, "num_machines": 2, "seed": 1}
        )
        assert len(random_inst) == 2

    def test_validate_instance_identifies_issues(self):
        """Invalid instances should surface detailed issues."""
        # Empty job, negative machine, zero duration, duplicate machine, missing machine 1
        bad_instance = [
            [
                (0, 3),
                (0, 0),
                (-1, 2),
            ],  # duplicate machine and zero duration and negative machine
            [],  # empty job
            [(2, 1)],  # leaves machine 1 missing overall
        ]
        is_valid, issues = validate_instance(bad_instance)

        assert not is_valid
        assert any("Job 0" in issue for issue in issues)
        assert any("empty" in issue.lower() for issue in issues)
        assert any("Missing machine IDs" in issue for issue in issues)

    def test_validate_instance_valid_case(self):
        """A well-formed instance passes validation."""
        instance = [[(0, 2), (1, 3)], [(1, 4), (0, 1)]]
        is_valid, issues = validate_instance(instance)
        assert is_valid
        assert issues == []

    def test_get_instance_info(self):
        """Summary info matches derived stats."""
        instance = _parse_instance(F3X3_INSTANCE)
        info = get_instance_info(instance)

        assert info["num_jobs"] == 3
        assert info["num_machines"] == 3
        assert info["num_operations"] == 9
        assert info["machines_used"] == [0, 1, 2]
        assert info["min_duration"] > 0
        assert info["max_duration"] >= info["min_duration"]

    def test_read_yaml_specification_instance(self):
        """YAML-like parser should decode jobs into machine-duration tuples."""
        text = "\njob0|(0,2) (1,3)\njob1|(1,4)\n"
        parsed = read_yaml_specification_instance(text)
        assert parsed == [[(0, 2), (1, 3)], [(1, 4)]]

    def test_read_yaml_specification_logistics(self):
        """Logistics YAML parser should return list of transport time tuples."""
        text = "\nm0|(1 2 3)\nm1|(4 5 6)\n"
        parsed = read_yaml_specification_logistics(text)
        assert parsed == [(1, 2, 3), (4, 5, 6)]

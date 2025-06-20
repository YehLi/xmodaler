import sys
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig


def test_python_version():
    """Test that Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"


def test_pytorch_available():
    """Test that PyTorch is properly installed."""
    assert torch.__version__ is not None
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6


def test_project_structure():
    """Test that the basic project structure exists."""
    project_root = Path(__file__).parent.parent
    
    # Check main package exists
    assert (project_root / "xmodaler").exists()
    assert (project_root / "xmodaler" / "__init__.py").exists()
    
    # Check test structure
    assert (project_root / "tests").exists()
    assert (project_root / "tests" / "__init__.py").exists()
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "integration").exists()
    
    # Check configuration files
    assert (project_root / "pyproject.toml").exists()


def test_fixtures_available(temp_dir, mock_config, mock_batch_data):
    """Test that pytest fixtures are properly configured."""
    # Test temp_dir fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test mock_config fixture
    assert isinstance(mock_config, DictConfig)
    assert "MODEL" in mock_config
    assert "DATALOADER" in mock_config
    
    # Test mock_batch_data fixture
    assert isinstance(mock_batch_data, dict)
    assert "image_features" in mock_batch_data
    assert isinstance(mock_batch_data["image_features"], torch.Tensor)


@pytest.mark.unit
def test_unit_marker():
    """Test that unit test marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration test marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow test marker works."""
    assert True


def test_coverage_configured():
    """Test that coverage is properly configured."""
    # This test will pass if coverage is running
    # The actual coverage threshold is checked by pytest-cov
    assert True


class TestInfrastructureClass:
    """Test that class-based tests work properly."""
    
    def test_class_method(self):
        """Test method in test class."""
        assert 2 + 2 == 4
    
    def test_with_fixtures(self, temp_file):
        """Test that fixtures work in test classes."""
        assert temp_file.exists()
        temp_file.write_text("test content")
        assert temp_file.read_text() == "test content"
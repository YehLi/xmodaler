import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Create a temporary file that is cleaned up after the test."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_config() -> DictConfig:
    """Provide a minimal mock configuration for testing."""
    config_dict = {
        "MODEL": {
            "DEVICE": "cpu",
            "META_ARCHITECTURE": "BaseEncoderDecoder",
            "ENCODER": "BaseEncoder",
            "DECODER": "BaseDecoder",
            "PREDICTOR": "BasePredictor",
            "VOCAB_SIZE": 10000,
            "MAX_SEQ_LEN": 100,
        },
        "DATALOADER": {
            "TRAIN_BATCH_SIZE": 4,
            "TEST_BATCH_SIZE": 4,
            "NUM_WORKERS": 0,
        },
        "SOLVER": {
            "BASE_LR": 0.001,
            "MAX_EPOCH": 10,
            "CHECKPOINT_PERIOD": 5,
        },
        "INPUT": {
            "DATASET": "MSCOCO",
            "TRAIN_SPLITS": ["train"],
            "VAL_SPLITS": ["val"],
            "TEST_SPLITS": ["test"],
        },
        "OUTPUT_DIR": "./output",
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_batch_data() -> Dict[str, Any]:
    """Provide mock batch data for testing."""
    batch_size = 4
    seq_len = 20
    feat_dim = 2048
    
    return {
        "image_features": torch.randn(batch_size, feat_dim),
        "input_ids": torch.randint(0, 10000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, 10000, (batch_size, seq_len)),
        "dataset_type": torch.zeros(batch_size, dtype=torch.long),
    }


@pytest.fixture
def cuda_available():
    """Check if CUDA is available and skip test if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_checkpoint_file(temp_dir: Path) -> Path:
    """Create a mock checkpoint file for testing."""
    checkpoint_path = temp_dir / "model_checkpoint.pth"
    checkpoint_data = {
        "model": {"dummy_weight": torch.randn(10, 10)},
        "optimizer": {"state": {}, "param_groups": []},
        "epoch": 5,
        "best_score": 0.95,
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_vocab_file(temp_dir: Path) -> Path:
    """Create a mock vocabulary file for testing."""
    vocab_path = temp_dir / "vocab.txt"
    vocab_content = "\n".join([
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "a",
        "cat",
        "dog",
        "image",
    ])
    vocab_path.write_text(vocab_content)
    return vocab_path


@pytest.fixture
def clean_environment():
    """Clean environment variables that might affect tests."""
    env_vars_to_clean = [
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
    ]
    
    original_env = {}
    for var in env_vars_to_clean:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]
    
    yield
    
    # Restore original environment
    for var, value in original_env.items():
        os.environ[var] = value


@pytest.fixture
def mock_image_tensor() -> torch.Tensor:
    """Create a mock image tensor (3, 224, 224)."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def mock_feature_extractor():
    """Mock feature extractor that returns fixed-size features."""
    class MockFeatureExtractor:
        def __call__(self, images):
            batch_size = images.size(0) if images.dim() > 3 else 1
            return torch.randn(batch_size, 2048)
    
    return MockFeatureExtractor()
"""Module contains fixtures needed fo test.

Author: Alexandru Mazurenco (2024)
License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path

from datasets import load_dataset
import hydra
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rnnt_lm_fusion import Workflow, set_seed


@pytest.fixture(scope="session")
def workflow(tmp_path_factory) -> Workflow:
    """
    Fixture to create a Workflow instance for session-wide use.

    Args:
        tmp_path_factory: A pytest fixture providing a factory for temporary directories.

    Returns:
        Workflow: An instance of the Workflow class configured for session-wide use.
    """
    config_path = "../conf"
    config_name = "config"

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, return_hydra_config=True)
        HydraConfig().cfg = cfg
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.root_params.log_dir = hydra_cfg.run.dir

        tmpdir = tmp_path_factory.mktemp("data")
        cfg.root_params.data_dir = Path(tmpdir).as_posix()
        OmegaConf.resolve(cfg)

    set_seed(cfg.root_params.seed)

    wf = Workflow(cfg)

    wf.cfg.librispeech = {
        "train": {"clean": "test"},
        "validation": {"clean": "test"},
        "test": {"clean": "test"},
    }

    wf.get_data()

    return wf


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def rescore_workflow(workflow: Workflow, request) -> Workflow:
    """
    Fixture to create a Workflow instance for session-wide use for rescoring.

    Args:
        workflow (Workflow): An instance of the Workflow class used as the base
        for rescoring operations.
        request: A pytest request object that provides information about the test context.

    Returns:
        Workflow: An instance of the Workflow class configured for session-wide use.
    """
    data = {
        "test_clean": load_dataset("librispeech_asr", "clean", split="test"),
    }
    workflow.cfg.root_params.device = request.param
    workflow.cfg.lstm.device = request.param
    workflow.get_eval_data_pool(data)
    return workflow

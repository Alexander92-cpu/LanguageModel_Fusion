"""
In this script we check that all basic training processes are working in this project.
"""

import os
from pathlib import Path
import pprint

import pytest
import torch

from rnnt_lm_fusion import Workflow

os.environ["WANDB_MODE"] = "disabled"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@pytest.mark.run(order=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_cuda_nums():
    """
    Check if number of available CUDAs is positive
    """
    assert torch.cuda.device_count() > 0, "No CUDA devices found."


class TestTrain:
    """
    Test class for training workflows.

    This class contains test methods related to training various models in the workflow.
    """
    @pytest.mark.run(order=1)
    def test_asr_tokenizer(self, workflow: Workflow):
        """
        Test the ASR tokenizer.

        This method tests the ASR tokenizer by ensuring it is not None and
        tokenizes a test sentence.

        Args:
            workflow (Workflow): The Workflow instance.

        Raises:
            AssertionError: If the tokenizer is None or tokenization fails.
        """
        assert workflow.lm_pool.asr_tokenizer is not None

        text = "Test sentence for tokenization"
        tokens = workflow.lm_pool.asr_tokenizer.tokenize(text)
        assert len(tokens) > 0

    @pytest.mark.run(order=2)
    def test_ngram_training(self, workflow: Workflow):
        """
        Test n-gram model training.

        This method trains the n-gram model and checks if the trained model file exists.

        Args:
            workflow (Workflow): The Workflow instance.

        Raises:
            AssertionError: If the trained model file does not exist.
        """
        workflow.train_ngram()
        assert Path(workflow.cfg.kenlm.model).exists()

    @pytest.mark.run(order=3)
    @pytest.mark.depends(on=["tests/test_main.py::test_cuda_nums"])
    def test_lstm_training(self, workflow: Workflow):
        """
        Test LSTM model training.

        This method trains the LSTM model and checks if the trained model file exists.
        It depends on the CUDA availability test.

        Args:
            workflow (Workflow): The Workflow instance.

        Raises:
            AssertionError: If the trained model file does not exist.
        """
        workflow.cfg.lstm.device = 'cuda'
        workflow.cfg.lstm.epochs = 1
        workflow.train_lstm()
        assert Path(workflow.cfg.lstm.save).exists()
        assert Path(workflow.cfg.lstm.tokenizer_path).exists()

    @pytest.mark.run(order=4)
    @pytest.mark.depends(on=["tests/test_main.py::test_cuda_nums"])
    def test_gpt2_training(self, workflow: Workflow):
        """
        Test GPT-2 model training.

        This method trains the GPT-2 model and checks if the trained model files exist.
        It depends on the CUDA availability test.

        Args:
            workflow (Workflow): The Workflow instance.

        Raises:
            AssertionError: If any of the trained model files do not exist.
        """
        workflow.cfg.gpt2.per_device_train_batch_size = 2
        workflow.train_gpt2()
        assert (Path(workflow.cfg.gpt2.dir_model) / "pytorch_model.bin").exists()
        assert (Path(workflow.cfg.gpt2.dir_model) / "training_args.bin").exists()
        assert (Path(workflow.cfg.gpt2.dir_model) / "config.json").exists()

    @pytest.mark.run(order=5)
    @pytest.mark.depends(on=["test_gpt2_training"])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_gpt2_evaluation(self, workflow: Workflow, device: str):
        """
        Test GPT-2 model evaluation.

        This method evaluates the GPT-2 model and checks if evaluation metrics
        are of the correct type.

        Args:
            workflow (Workflow): The Workflow instance.
            device (str): The device for evaluation.

        Raises:
            AssertionError: If evaluation metrics are not of the correct type.
        """
        workflow.cfg.root_params.device = device
        baseline = workflow.evaluate_gpt2()
        pprint.pprint(baseline)
        assert isinstance(baseline["eval_loss"], float)
        assert isinstance(baseline["perplexity"], float)


class TestRescore:
    """
    Test class for rescoring workflows.

    This class contains test methods related to the rescoring workflow.
    """
    @pytest.mark.run(order=6)
    @pytest.mark.depends(on=["tests/test_main.py::TestTrain::test_ngram_training",
                             "tests/test_main.py::TestTrain::test_lstm_training",
                             "tests/test_main.py::TestTrain::test_gpt2_training"])
    def test_rescoring(self, rescore_workflow: Workflow):
        """
        Test rescoring.

        This method performs rescoring and checks if the required outputs are present.

        Args:
            rescore_workflow (Workflow): The rescore Workflow instance.

        Raises:
            AssertionError: If required outputs are not present.
        """
        rescore_workflow.cfg.rescore.methods.baseline = True
        rescore_workflow.cfg.rescore.methods.lodr = True
        rescore_workflow.cfg.rescore.methods.dr = True
        rescore_workflow.cfg.rescore.methods.ilme = True
        rescore_workflow.cfg.rescore.methods.sf = True
        rescore_workflow.cfg.rescore.num_steps = 10

        info = rescore_workflow.rescore()
        pprint.pprint(info)
        assert "outputs" in info["test_clean"]
        assert "references" in info["test_clean"]


class TestOptimize:
    """
    Test class for optimization workflows.

    This class contains test methods related to the optimization workflow.
    """
    @pytest.mark.run(order=7)
    @pytest.mark.depends(on=["tests/test_main.py::TestTrain::test_ngram_training",
                             "tests/test_main.py::TestTrain::test_lstm_training",
                             "tests/test_main.py::TestTrain::test_gpt2_training"])
    def test_optimization(self, rescore_workflow: Workflow):
        """
        Test optimization.

        This method performs optimization and checks if the database export exists.

        Args:
            rescore_workflow (Workflow): The rescore Workflow instance.

        Raises:
            AssertionError: If the database export does not exist.
        """
        rescore_workflow.cfg.rescore.methods.baseline = True
        rescore_workflow.cfg.rescore.methods.lodr = True
        rescore_workflow.cfg.rescore.methods.dr = True
        rescore_workflow.cfg.rescore.methods.ilme = True
        rescore_workflow.cfg.rescore.methods.sf = True
        rescore_workflow.cfg.rescore.num_steps = 10
        rescore_workflow.cfg.optimize.n_trials = 10

        info = rescore_workflow.rescore()
        pprint.pprint(info)
        rescore_workflow.optimize_hyperparams(info)
        assert Path(rescore_workflow.cfg.optimize.db_exp).exists()

"""
Unit tests for the Experiment 3 script.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from run_experiment3_sensitivity import run_experiment_3


@pytest.fixture
def mock_model_and_data():
    """Mock the model and data loader to avoid actual computation."""
    with (
        patch("run_experiment3_sensitivity.load_vit_model") as mock_load_model,
        patch(
            "run_experiment3_sensitivity.create_imagenet_val_loader"
        ) as mock_create_loader,
        patch("run_experiment3_sensitivity.get_linear_layers") as mock_get_layers,
        patch("run_experiment3_sensitivity.evaluate_top1_accuracy") as mock_evaluate,
        patch("builtins.open"),
        patch("csv.DictWriter"),
        patch("run_experiment3_sensitivity.visualize_results") as mock_visualize,
    ):
        mock_model = MagicMock()
        mock_load_model.return_value = (mock_model, MagicMock())

        # Configure the layer mock to have a tensor attribute for weight.data
        mock_layer = MagicMock()
        mock_layer.weight.data = torch.randn(1, 1)
        mock_get_layers.return_value = {"test_layer": mock_layer}

        mock_evaluate.return_value = 80.0

        yield mock_create_loader, mock_visualize


def test_experiment_3_runs_with_num_images(mock_model_and_data):
    """Verify that the experiment runs with a specific number of images."""
    mock_create_loader, mock_visualize = mock_model_and_data
    run_experiment_3(num_images=128)
    # Verify that the data loader was called with the correct number of images
    mock_create_loader.assert_called_with(batch_size=64, max_images=128)
    # Verify that the visualizer was called
    mock_visualize.assert_called_once()


def test_experiment_3_runs_with_all_images(mock_model_and_data):
    """Verify that the experiment runs with all images."""
    mock_create_loader, mock_visualize = mock_model_and_data
    run_experiment_3(num_images=None)
    # Verify that the data loader was called with max_images=None
    mock_create_loader.assert_called_with(batch_size=64, max_images=None)
    # Verify that the visualizer was called
    mock_visualize.assert_called_once()

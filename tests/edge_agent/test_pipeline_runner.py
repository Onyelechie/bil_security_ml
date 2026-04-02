import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from edge_agent.pipeline_runner import PipelineRunner
from edge_agent.video.ring_buffer import FrameItem


@pytest.fixture(autouse=True)
def _no_disk_writes(mocker):
    return mocker.patch("edge_agent.pipeline_runner.cv2.imwrite", return_value=True)


def test_pipeline_runner_uses_frameitem_timestamp(tmp_path):
    evaluator = MagicMock()
    evaluator.evaluate_frames.return_value = {
        "detection": {"label": "person", "confidence": 0.9},
        "frame": np.zeros((10, 10, 3), dtype=np.uint8),
        "frame_index": 1,
    }

    sender = MagicMock()
    sender.send_alert.return_value = True

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=str(tmp_path),
    )

    ts0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

    frames = [
        FrameItem(ts=ts0, frame=np.zeros((10, 10), dtype=np.uint8)),
        FrameItem(ts=ts1, frame=np.zeros((10, 10), dtype=np.uint8)),
    ]

    pipeline.process_frames("cam-1", frames)

    sender.send_alert.assert_called_once()
    _, kwargs = sender.send_alert.call_args
    assert kwargs["timestamp"] == ts1


def test_save_frame_returns_absolute_path(tmp_path):
    evaluator = MagicMock()
    sender = MagicMock()

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=str(tmp_path),
    )

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    path = pipeline._save_frame("cam-1", frame)

    assert path is not None
    assert os.path.isabs(path)


def test_pipeline_runner_skips_on_no_detection(tmp_path):
    evaluator = MagicMock()
    evaluator.evaluate_frames.return_value = None

    sender = MagicMock()

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=str(tmp_path),
    )

    frames = [np.zeros((10, 10), dtype=np.uint8)]

    pipeline.process_frames("cam-1", frames)

    sender.send_alert.assert_not_called()


def test_pipeline_runner_save_images_disabled(tmp_path):
    evaluator = MagicMock()
    evaluator.evaluate_frames.return_value = {
        "detection": {"label": "person", "confidence": 0.9},
        "frame": np.zeros((10, 10, 3), dtype=np.uint8),
        "frame_index": 0,
    }

    sender = MagicMock()
    sender.send_alert.return_value = True

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=str(tmp_path),
        save_images=False,
    )

    frames = [np.zeros((10, 10), dtype=np.uint8)]

    pipeline.process_frames("cam-1", frames)
    _, kwargs = sender.send_alert.call_args
    assert kwargs["image_path"] is None


def test_pipeline_runner_rejects_invalid_frames(tmp_path):
    evaluator = MagicMock()
    sender = MagicMock()

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=str(tmp_path),
    )

    with pytest.raises(ValueError):
        pipeline.process_frames("cam-1", np.zeros((10, 10), dtype=np.uint8))

    with pytest.raises(ValueError):
        pipeline.process_frames("cam-1", [object()])


def test_save_frame_avoids_filename_collisions(tmp_path, mocker):
    evaluator = MagicMock()
    sender = MagicMock()

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=str(tmp_path),
    )

    fixed_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    mock_dt = mocker.patch("edge_agent.pipeline_runner.datetime")
    mock_dt.now.return_value = fixed_ts

    u1 = MagicMock()
    u1.hex = "aaaabbbb"
    u2 = MagicMock()
    u2.hex = "ccccdddd"
    mocker.patch("edge_agent.pipeline_runner.uuid4", side_effect=[u1, u2])

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    path1 = pipeline._save_frame("cam-1", frame)
    path2 = pipeline._save_frame("cam-1", frame)

    assert path1 != path2

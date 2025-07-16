"""
Checkpoint Utilities Module

This module provides utility functions for managing and locating checkpoint files during training and testing.
It includes functions to find specific checkpoint files based on user input or criteria such as the latest or best checkpoint.

Functions:
    - find_checkpoint_file: Locates a checkpoint file based on user input or criteria.
    - find_resume_checkpoint: Finds a checkpoint file for resuming training.
    - find_test_checkpoint: Finds a checkpoint file for testing a model.

Dependencies:
    - glob: For pattern matching file paths.
    - os: For file system operations.
    - pathlib.Path: For handling file paths.
    - atria_core.logger.logger: For logging messages.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from atria_core.logger.logger import get_logger

logger = get_logger(__name__)


def find_checkpoint_file(
    filename, checkpoint_dir: str, load_best: bool = False, resume=True, quiet=False
):
    """
    Locates a checkpoint file based on user input or criteria such as the latest or best checkpoint.

    Args:
        filename (str): The name of the checkpoint file to locate. If None, the function searches for the latest or best checkpoint.
        checkpoint_dir (str): The directory where checkpoint files are stored.
        load_best (bool, optional): Whether to prioritize finding the best checkpoint. Defaults to False.
        resume (bool, optional): Whether the checkpoint is for resuming training. Defaults to True.
        quiet (bool, optional): Whether to suppress logging messages. Defaults to False.

    Returns:
        pathlib.Path or None: The path to the located checkpoint file, or None if no file is found.

    Notes:
        - If `filename` is provided and exists, it is returned directly.
        - If no `filename` is provided, the function searches for checkpoint files in `checkpoint_dir`.
        - If `load_best` is True, only checkpoints containing "best" in their name are considered.
        - If `resume` is True, logs a message about resuming training from the located checkpoint.
    """
    import glob
    import os
    from pathlib import Path

    if not checkpoint_dir.exists():
        return

    if filename is not None:
        if Path(filename).exists():
            return Path(filename)
        elif Path(checkpoint_dir / filename).exists():
            return Path(checkpoint_dir / filename)
        else:
            logger.warning(
                f"User provided checkpoint file filename={filename} not found."
            )

    list_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")
    if len(list_checkpoints) > 0:
        if not load_best:
            list_checkpoints = [c for c in list_checkpoints if "best" not in c]
        else:
            list_checkpoints = [c for c in list_checkpoints if "best" in c]

        if len(list_checkpoints) > 0:
            latest_checkpoint = max(list_checkpoints, key=os.path.getctime)
            if resume:
                if not quiet:
                    logger.info(
                        f"Checkpoint detected, resuming training from {latest_checkpoint}. To avoid this behavior, change "
                        "the `output_dir` or add `overwrite_output_dir` to train from scratch."
                    )
            else:
                if not quiet:
                    logger.info(
                        f"Checkpoint detected, testing model using checkpoint {latest_checkpoint}."
                    )
            return latest_checkpoint


def find_resume_checkpoint(
    resume_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False
):
    """
    Finds a checkpoint file for resuming training.

    Args:
        resume_checkpoint_file (str): The name of the checkpoint file to locate for resuming training.
        checkpoint_dir (str): The directory where checkpoint files are stored.
        load_best (bool, optional): Whether to prioritize finding the best checkpoint. Defaults to False.

    Returns:
        pathlib.Path or None: The path to the located checkpoint file, or None if no file is found.

    Notes:
        - This function is a wrapper around `find_checkpoint_file` with `resume` set to True.
    """
    return find_checkpoint_file(
        filename=resume_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=True,
    )


def find_test_checkpoint(
    test_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False
):
    """
    Finds a checkpoint file for testing a model.

    Args:
        test_checkpoint_file (str): The name of the checkpoint file to locate for testing.
        checkpoint_dir (str): The directory where checkpoint files are stored.
        load_best (bool, optional): Whether to prioritize finding the best checkpoint. Defaults to False.

    Returns:
        pathlib.Path or None: The path to the located checkpoint file, or None if no file is found.

    Notes:
        - This function is a wrapper around `find_checkpoint_file` with `resume` set to False.
    """
    return find_checkpoint_file(
        filename=test_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=False,
    )

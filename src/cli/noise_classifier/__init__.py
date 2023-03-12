"""
Noise Classifier CLI
-------------------

This module contains the CLI for the noise classifier, which is used to classify Audio Blocks and produce the Irregularity File.

"""

from .classifier import classify, write_irregularity_file

__all__ = ["classify", "write_irregularity_file"]

"""Layer 4: dual white-box watermarking (visible image + invisible bits)."""

from qsfl.watermark.embedder import DualWatermarker, WatermarkBundle
from qsfl.watermark.uchida import extract_bits, embed_bits

__all__ = ["DualWatermarker", "WatermarkBundle", "embed_bits", "extract_bits"]

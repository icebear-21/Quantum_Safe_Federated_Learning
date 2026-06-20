"""Build the watermark payloads: primary visible image P, secondary bits S."""

from __future__ import annotations

import numpy as np

PRIMARY_SIZE = 16  # primary watermark rendered to a 16x16 binary image (256 bits)


def make_secondary_bits(length: int, payload_seed: int) -> np.ndarray:
    """Deterministic invisible bit-sequence ``S`` of given length."""
    rng = np.random.default_rng(payload_seed)
    return rng.integers(0, 2, size=length).astype(np.int64)


def make_primary_image(primary_cfg, size: int = PRIMARY_SIZE) -> np.ndarray:
    """Render the primary watermark to a ``size x size`` binary image (1=ink).

    ``type=qr`` renders a QR of the payload; ``type=image`` loads an image file.
    The result is a binary pattern used for NC comparison (we never decode it).
    """
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError("pillow is required for the primary watermark image.") from exc

    wtype = str(primary_cfg.type).lower()
    if wtype == "qr":
        try:
            import qrcode
        except Exception as exc:  # pragma: no cover
            raise ImportError("qrcode is required for type=qr watermarks.") from exc
        qr = qrcode.QRCode(border=1, box_size=2)
        qr.add_data(str(primary_cfg.payload))
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("L")
    elif wtype == "image":
        if not primary_cfg.get("image_path"):
            raise ValueError("watermark.primary.image_path is required for type=image")
        img = Image.open(str(primary_cfg.image_path)).convert("L")
    else:
        raise ValueError(f"unknown primary watermark type: {wtype}")

    img = img.resize((size, size), Image.NEAREST)
    arr = np.asarray(img)
    return (arr < 128).astype(np.int64)  # ink (dark) -> 1

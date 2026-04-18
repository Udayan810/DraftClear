import numpy as np
import pytest

from cad_converter import CADConverter, DWGNotSupportedError
from pdf_compiler import PDFCompiler


def test_cad_converter_normalizes_grayscale_to_bgr_uint8():
    grayscale = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    normalized = CADConverter._normalize_image(grayscale)

    assert normalized.dtype == np.uint8
    assert normalized.shape == (2, 2, 3)
    assert normalized[0, 1, 0] == 127


def test_pdf_compiler_normalizes_bgra_to_bgr():
    bgra = np.zeros((3, 3, 4), dtype=np.uint8)
    bgra[:, :, 2] = 255
    bgra[:, :, 3] = 255

    normalized = PDFCompiler._normalize_image(bgra)

    assert normalized.shape == (3, 3, 3)
    assert np.all(normalized[:, :, 2] == 255)


def test_convert_dwg_to_dxf_bytes_requires_local_converter(monkeypatch):
    monkeypatch.setattr(CADConverter, "_find_oda_converter", classmethod(lambda cls: None))
    monkeypatch.setattr(CADConverter, "_convert_dwg_with_autocad", classmethod(lambda cls, dwg_path, dxf_path: None))

    with pytest.raises(DWGNotSupportedError) as exc_info:
        CADConverter.convert_dwg_to_dxf_bytes(b"dummy-dwg-content")

    assert "convert internally" in str(exc_info.value)

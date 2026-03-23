"""Real implementation of ModelScanner — scans a folder for video model files."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from api_types import DetectedModel

# GGUF magic bytes (4-byte little-endian magic + 4-byte version)
_GGUF_MAGIC = b"GGUF"
_GGUF_MIN_FILE_SIZE = 8  # magic (4) + version (4)

# Safetensors header sentinel (little-endian uint64 for header length)
_SAFETENSORS_HEADER_LENGTH_BYTES = 8


class ModelScannerImpl:
    """Scans a directory for video model files using file metadata (not size heuristics)."""

    def scan_video_models(self, folder: Path) -> list[DetectedModel]:
        """Return all detected video models in *folder*. Returns [] if folder doesn't exist."""
        if not folder.exists() or not folder.is_dir():
            return []

        results: list[DetectedModel] = []

        for entry in sorted(folder.iterdir()):
            try:
                if entry.is_file():
                    if entry.suffix.lower() == ".gguf":
                        model = self._scan_gguf(entry)
                        if model is not None:
                            results.append(model)
                    elif entry.suffix.lower() == ".safetensors":
                        model = self._scan_safetensors(entry)
                        if model is not None:
                            results.append(model)
                elif entry.is_dir():
                    model = self._scan_nf4_folder(entry)
                    if model is not None:
                        results.append(model)
            except Exception:
                continue  # skip inaccessible entries

        return results

    # ------------------------------------------------------------------
    # GGUF
    # ------------------------------------------------------------------

    def _scan_gguf(self, path: Path) -> DetectedModel | None:
        """Return a DetectedModel if *path* is a valid GGUF file, else None."""
        try:
            with path.open("rb") as f:
                header = f.read(_GGUF_MIN_FILE_SIZE)
            if len(header) < _GGUF_MIN_FILE_SIZE:
                return None
            magic = header[:4]
            if magic != _GGUF_MAGIC:
                return None
            version = struct.unpack_from("<I", header, 4)[0]
            if version < 1:
                return None
        except OSError:
            return None

        stat = path.stat()
        size_bytes = stat.st_size
        size_gb = round(size_bytes / (1024**3), 2)
        quant_type = self._quant_type_from_filename(path.name)

        return DetectedModel(
            filename=path.name,
            path=str(path),
            model_format="gguf",
            quant_type=quant_type,
            size_bytes=size_bytes,
            size_gb=size_gb,
            is_distilled=False,
            display_name=self._gguf_display_name(path.name, quant_type),
        )

    def _quant_type_from_filename(self, filename: str) -> str | None:
        """Extract quant type like Q8_0, Q5_K_M, Q4_K_M from a GGUF filename."""
        name_upper = filename.upper()
        # Common GGUF quant suffixes ordered from most to least specific
        candidates = [
            "Q8_0", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
            "Q3_K_M", "Q3_K_S", "Q2_K", "F16", "F32",
        ]
        for cand in candidates:
            if cand in name_upper:
                return cand
        return None

    def _gguf_display_name(self, filename: str, quant_type: str | None) -> str:
        stem = Path(filename).stem
        if quant_type:
            return f"{stem} ({quant_type})"
        return stem

    # ------------------------------------------------------------------
    # Safetensors
    # ------------------------------------------------------------------

    def _scan_safetensors(self, path: Path) -> DetectedModel | None:
        """Return a DetectedModel if *path* is a valid .safetensors video model, else None."""
        try:
            with path.open("rb") as f:
                raw_len = f.read(_SAFETENSORS_HEADER_LENGTH_BYTES)
            if len(raw_len) < _SAFETENSORS_HEADER_LENGTH_BYTES:
                return None
        except OSError:
            return None

        fmt = self._detect_safetensors_format(path)
        stat = path.stat()
        size_bytes = stat.st_size
        size_gb = round(size_bytes / (1024**3), 2)

        return DetectedModel(
            filename=path.name,
            path=str(path),
            model_format=fmt,
            quant_type=None,
            size_bytes=size_bytes,
            size_gb=size_gb,
            is_distilled=False,
            display_name=path.stem,
        )

    def _detect_safetensors_format(self, path: Path) -> str:
        """Determine bf16 vs fp8 by inspecting companion config.json or safetensors header."""
        # Check sibling config.json for torch_dtype
        config_path = path.parent / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                dtype = str(data.get("torch_dtype", "")).lower()
                if "fp8" in dtype or "float8" in dtype:
                    return "fp8"
                if "bf16" in dtype or "bfloat16" in dtype:
                    return "bf16"
            except (OSError, json.JSONDecodeError):
                pass

        # Fall back to checking the safetensors header for dtype strings
        try:
            with path.open("rb") as f:
                raw_len = f.read(8)
                if len(raw_len) < 8:
                    return "bf16"
                header_len = struct.unpack_from("<Q", raw_len)[0]
                # Cap header read at 1 MB to avoid blowing memory on corrupt files
                header_len = min(header_len, 1024 * 1024)
                header_bytes = f.read(header_len)
            header_text = header_bytes.decode("utf-8", errors="replace")
            if "F8" in header_text or "float8" in header_text.lower():
                return "fp8"
        except OSError:
            pass

        return "bf16"

    # ------------------------------------------------------------------
    # NF4 folder
    # ------------------------------------------------------------------

    def _scan_nf4_folder(self, folder: Path) -> DetectedModel | None:
        """Return a DetectedModel if *folder* contains a quantize_config.json with quant_type=nf4."""
        config_path = folder / "quantize_config.json"
        if not config_path.exists():
            return None

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        quant_type = str(data.get("quant_type", "")).lower()
        if quant_type != "nf4":
            return None

        # Sum up all files in the folder for size
        size_bytes = sum(
            f.stat().st_size
            for f in folder.rglob("*")
            if f.is_file()
        )
        size_gb = round(size_bytes / (1024**3), 2)

        return DetectedModel(
            filename=folder.name,
            path=str(folder),
            model_format="nf4",
            quant_type="nf4",
            size_bytes=size_bytes,
            size_gb=size_gb,
            is_distilled=False,
            display_name=folder.name,
        )

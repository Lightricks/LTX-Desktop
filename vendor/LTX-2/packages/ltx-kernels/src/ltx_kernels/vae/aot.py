"""Build and load the production DiffVAE CuTe DSL AOT bundle."""

from __future__ import annotations

import functools
import hashlib
import importlib.resources
import os
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cutlass
import cutlass.cute as cute
import torch

CompiledFunction = Callable[..., int | None]
CompileKernel = Callable[..., Any]

_PACKAGED_BUNDLE_DIRECTORY = ("_aot", "sm100a")
_PACKAGED_GPU_ARCH = "sm_100a"


@dataclass(frozen=True)
class KernelKey:
    """One architecture-specific specialization, used for cache and object identity."""

    compile_kernel: CompileKernel
    gpu_arch: str
    params: tuple[tuple[str, Hashable], ...]

    @classmethod
    def create(
        cls,
        compile_kernel: CompileKernel,
        gpu_arch: str,
        **params: Hashable,
    ) -> "KernelKey":
        for name, value in params.items():
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError(f"compile-time parameter {name!r} must be hashable, got {value!r}") from exc
        return cls(compile_kernel, gpu_arch, tuple(sorted(params.items())))

    @property
    def kind(self) -> str:
        return self.compile_kernel.__module__.rsplit(".", maxsplit=1)[-1].removesuffix("_dsl")

    @property
    def name(self) -> str:
        identity = (
            self.compile_kernel.__module__,
            self.compile_kernel.__qualname__,
            self.gpu_arch,
            self.params,
        )
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
        return f"{self.kind}_{self.gpu_arch}_{digest}"

    @property
    def function_prefix(self) -> str:
        return f"ltx_vae_{self.name}"

    @property
    def compile_kwargs(self) -> dict[str, Hashable]:
        return dict(self.params)


def _device_and_arch(device: torch.device) -> tuple[torch.device, str]:
    if device.type != "cuda":
        raise ValueError(f"VAE DSL kernels need a CUDA device, got {device}")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    target_device = torch.device("cuda", device_index)
    major, minor = torch.cuda.get_device_capability(target_device)
    return target_device, f"sm_{major}{minor}a"


def fake_dynamic_vector(dtype: type[cutlass.Numeric]) -> cute.Tensor:
    """Match ``from_dlpack(vector).mark_layout_dynamic()`` without storage."""
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int32(),),
        (1,),
    )


def fake_dynamic_matrix(dtype: type[cutlass.Numeric], *, assumed_align: int) -> cute.Tensor:
    """Match the arbitrary row stride accepted by ``_dyn_act``."""
    rows = cute.sym_int32()
    columns = cute.sym_int32()
    row_stride = cute.sym_int64()
    return cute.runtime.make_fake_tensor(
        dtype,
        (rows, columns),
        (row_stride, 1),
        assumed_align=assumed_align,
    )


def fake_dynamic_weight(dtype: type[cutlass.Numeric], *, columns: int) -> cute.Tensor:
    """Match ``_dyn_w``: compact rows with a dynamically typed K dimension."""
    rows = cute.sym_int32()
    dynamic_columns = cute.sym_int32(divisibility=columns)
    row_stride = cute.sym_int64(divisibility=columns)
    return cute.runtime.make_fake_tensor(
        dtype,
        (rows, dynamic_columns),
        (row_stride, 1),
        assumed_align=32,
    )


def _bundle_root() -> importlib.resources.abc.Traversable:
    root = importlib.resources.files("ltx_kernels.vae")
    for part in _PACKAGED_BUNDLE_DIRECTORY:
        root = root.joinpath(part)
    return root


@functools.lru_cache(maxsize=1)
def _bundle_is_packaged() -> bool:
    return _bundle_root().is_dir()


@functools.lru_cache(maxsize=16)
def _load_function(file_name: str, function_prefix: str) -> CompiledFunction:
    resource = _bundle_root().joinpath(file_name)
    if not resource.is_file():
        raise RuntimeError(f"Packaged VAE DSL AOT object is missing: {file_name}")
    try:
        with importlib.resources.as_file(resource) as object_path:
            module = cute.runtime.load_module(os.fspath(object_path))
        return cast(CompiledFunction, getattr(module, function_prefix))
    except Exception as exc:
        raise RuntimeError(f"Failed to load packaged VAE DSL AOT object {file_name}; refusing JIT fallback") from exc


@functools.lru_cache(maxsize=8)
def production_kernel_keys(gpu_arch: str) -> tuple[KernelKey, ...]:
    """Return the finite production specialization set for ``gpu_arch``."""
    from ltx_kernels.vae import block_fna_dsl, na_attn_dsl, rope_dsl  # noqa: PLC0415

    na = na_attn_dsl._compile_kernel
    block = block_fna_dsl._compile_kernel
    rope = rope_dsl._compile_kernel
    return (
        KernelKey.create(
            na,
            gpu_arch,
            kernel_size=(3, 7, 7),
            tile_thw=(2, 8, 8),
            keyframes=False,
            keyframe_queries=False,
        ),
        KernelKey.create(
            na,
            gpu_arch,
            kernel_size=(3, 7, 7),
            tile_thw=(2, 8, 8),
            keyframes=True,
            keyframe_queries=True,
        ),
        KernelKey.create(
            na,
            gpu_arch,
            kernel_size=(3, 5, 5),
            tile_thw=(2, 8, 8),
            keyframes=False,
            keyframe_queries=False,
        ),
        KernelKey.create(
            na,
            gpu_arch,
            kernel_size=(3, 5, 5),
            tile_thw=(2, 8, 8),
            keyframes=True,
            keyframe_queries=True,
        ),
        KernelKey.create(
            block,
            gpu_arch,
            C=256,
            Ctx=512,
            upsample_stride=(2, 2, 2),
            Hidd=1024,
            num_heads=4,
            rope_dim_split=(16, 24, 24),
            kernel_size=(11, 11, 11),
            tile_thw=(4, 4, 8),
            slab_box_max=block_fna_dsl.SLAB_BOX_MAX,
            keyframes=False,
        ),
        KernelKey.create(
            block,
            gpu_arch,
            C=256,
            Ctx=512,
            upsample_stride=(2, 2, 2),
            Hidd=1024,
            num_heads=4,
            rope_dim_split=(16, 24, 24),
            kernel_size=(11, 11, 11),
            tile_thw=(4, 4, 8),
            slab_box_max=block_fna_dsl.SLAB_BOX_MAX,
            keyframes=True,
        ),
        KernelKey.create(
            rope,
            gpu_arch,
            rope_dims=(16, 24, 24),
        ),
    )


class KernelCache:
    """Compile or load VAE DSL kernels under one canonical key."""

    def __init__(self) -> None:
        self._compiled: dict[KernelKey, CompiledFunction] = {}

    def get(
        self,
        compile_kernel: CompileKernel,
        device: torch.device,
        **params: Hashable,
    ) -> CompiledFunction:
        target_device, gpu_arch = _device_and_arch(device)
        key = KernelKey.create(compile_kernel, gpu_arch, **params)
        if key in self._compiled:
            return self._compiled[key]

        with torch.cuda.device(target_device):
            compiled = self._load_precompiled(key)
            if compiled is None:
                compiled = self.compile(key, options=os.environ.get("CUTE_DSL_OPTS"))
            self._compiled[key] = compiled
        return compiled

    @staticmethod
    def compile(key: KernelKey, options: str | None = None) -> CompiledFunction:
        return cast(
            CompiledFunction,
            key.compile_kernel(
                **key.compile_kwargs,
                **({"options": options} if options else {}),
            ),
        )

    @staticmethod
    def _load_precompiled(key: KernelKey) -> CompiledFunction | None:
        if key.gpu_arch != _PACKAGED_GPU_ARCH:
            return None
        if key not in production_kernel_keys(key.gpu_arch):
            return None
        if not _bundle_is_packaged():
            return None
        return _load_function(f"{key.name}.o", key.function_prefix)

    def count(self, compile_kernel: CompileKernel) -> int:
        return sum(key.compile_kernel is compile_kernel for key in self._compiled)

    def clear(self, compile_kernel: CompileKernel | None = None) -> None:
        if compile_kernel is None:
            self._compiled.clear()
            return
        for key in tuple(self._compiled):
            if key.compile_kernel is compile_kernel:
                del self._compiled[key]


KERNEL_CACHE = KernelCache()


def export_production_artifacts(
    output_directory: Path,
    *,
    gpu_arch: str,
    host_target: str,
) -> None:
    """Compile and export every production key into ``output_directory``."""
    if gpu_arch != _PACKAGED_GPU_ARCH:
        raise ValueError(f"Production VAE AOT bundle targets {_PACKAGED_GPU_ARCH}, got {gpu_arch}")
    output_directory.mkdir(parents=True, exist_ok=True)
    options = f"--gpu-arch {gpu_arch}"
    if host_target:
        options = f"{options} --host-target {host_target}"
    if extra_options := os.environ.get("CUTE_DSL_OPTS"):
        options = f"{options} {extra_options}"

    for key in production_kernel_keys(gpu_arch):
        compiled = KERNEL_CACHE.compile(key, options=options)
        compiled.export_to_c(
            file_path=os.fspath(output_directory),
            file_name=key.name,
            function_prefix=key.function_prefix,
        )
        output_directory.joinpath(f"{key.name}.h").unlink()

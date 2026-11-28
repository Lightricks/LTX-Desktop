"""Origin-agnostic pairing/session helpers for remote routes."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from _routes._errors import HTTPError
from api_types import Asset, Generation
from remote.dto import RemoteAsset, RemoteGeneration, to_remote_asset, to_remote_generation
from remote.media_urls import MediaUrlSigner
from remote.pairing import AuthenticatedDevice, RemotePairing


@dataclass(frozen=True)
class RemoteContext:
    signer: MediaUrlSigner
    device: AuthenticatedDevice

    def asset(self, asset: Asset) -> RemoteAsset:
        return to_remote_asset(asset, signer=self.signer, device_id=self.device.id)

    def generation(self, generation: Generation) -> RemoteGeneration:
        return to_remote_generation(
            generation, signer=self.signer, device_id=self.device.id
        )


def get_remote_pairing(request: Request) -> RemotePairing:
    pairing = getattr(request.app.state, "remote_pairing", None)
    if not isinstance(pairing, RemotePairing):
        raise HTTPError(500, "PAIRING_UNAVAILABLE")
    return pairing


def get_paired_device(request: Request) -> AuthenticatedDevice:
    device = getattr(request.state, "paired_device", None)
    if not isinstance(device, AuthenticatedDevice):
        raise HTTPError(401, "Unauthorized")
    return device


def get_remote_context(request: Request) -> RemoteContext:
    return RemoteContext(
        signer=get_remote_pairing(request).signer,
        device=get_paired_device(request),
    )

"""Stage the app-local x64 VC14 runtime used by LTX Desktop - AMD 395 MAX.

Accept the exact versioned bootstrap artifact identified in bootstrap-sources.json.
The signed payload contains nested cabinets. Verify the embedded CRT files individually.
"""
from __future__ import annotations

import hashlib
import struct
import subprocess
import tempfile
from pathlib import Path

REDIST_HASH = '9c02c9ed120f671b900e6b9471d890e3b2414117a7e53f4ae2e855466fa2b7c9'
CRT_DLLS = {
    'msvcp140.dll': '59d325227321d52b9a523ea401ce6787a5d59f78229a0ff3a6f124e5d3445685',
    'msvcp140_atomic_wait.dll': '8b10c6bd4c3c427ce7c8be96ef0fef0f8187bc236962be62a03caef834b37dcd',
    'vcruntime140.dll': 'eb1ee979bf02324712346e5bdbfc59c785388bc50d18cc2688aae4b0734915a0',
    'vcruntime140_1.dll': 'a14b4a42a2c6df04d6d53a8340816023d953406f7f3d94996c7cab30c414ff63',
}

def extract(redist: bytes, output: Path, sevenzip: str) -> None:
    if hashlib.sha256(redist).hexdigest() != REDIST_HASH:
        raise ValueError('This build requires the Microsoft VC++ v14.52.37029.0 x64 redistributable.')
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # The pinned WiX Burn bundle contains two adjacent CAB containers. Its
        # attached data CAB contains the packaged x64 CRT CAB, identified as a4.
        offset = redist.index(b'MSCF\0\0\0\0', redist.index(b'MSCF\0\0\0\0') + 8)
        size = struct.unpack_from('<I', redist, offset + 8)[0]
        cab = work / 'attached.cab'
        cab.write_bytes(redist[offset:offset+size])
        for source in (cab, work / 'a4'):
            subprocess.run([sevenzip,'x','-y','-o'+str(work),str(source)],check=True,stdout=subprocess.DEVNULL)
        output.mkdir(exist_ok=True, parents=True)
        for dll, expected in CRT_DLLS.items():
            data = (work / (dll+'_amd64')).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Microsoft CRT SHA-256 mismatch for {dll}')
            (output / dll).write_bytes(data)

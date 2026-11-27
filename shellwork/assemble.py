"""Assemble the Windows Electron app. The ROCm runtime is staged separately."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

work = Path(os.environ.get("LTX_SHELL_WORK", Path(__file__).resolve().parent)).resolve()
app = Path(sys.argv[1]).resolve()
release = work / 'win-x64'
resources = release / 'resources'
container = work / 'asar-input'
expected_electron_sha256 = 'dcd36396a606a5ae2f5651b4ee6bb463a624dbf15f786eda57cee2cc361c138c'
expected_koffi_sha512 = 'BPeQXc1bRd0QBOklvsP+AjoRnUzKbPNE6rfx7VNxrebhh09MKld2ibstgKWn6ejQLEcfKEoUJ+WAWIhX4AOsIg=='

def verify(file, algorithm, expected):
    with file.open('rb') as stream:
        got = hashlib.file_digest(stream, algorithm).digest()
    if got != expected:
        raise ValueError(f'{file.name}: unexpected {algorithm}')

verify(work/'electron.zip', 'sha256', bytes.fromhex(expected_electron_sha256))
verify(work/'koffi-win32-x64.tgz', 'sha512', base64.b64decode(expected_koffi_sha512))
upstream_manifest = json.loads((app / 'package.json').read_text(encoding='utf-8'))
product_name = upstream_manifest['productName']
launcher = release / (product_name + '.exe')
old_launcher = release / 'LTX Desktop AMD.exe'
refresh_only = '--refresh-app' in sys.argv[2:]
if refresh_only:
    if not (launcher.is_file() or old_launcher.is_file()):
        raise RuntimeError('Build the Windows payload before refreshing app resources.')
else:
    if (resources/'python').exists():
        raise RuntimeError('Use --refresh-app to retain the staged AMD runtime.')
    if release.exists():
        shutil.rmtree(release)
    release.mkdir()
    with zipfile.ZipFile(work/'electron.zip') as archive:
        archive.extractall(release)
    (resources/'default_app.asar').unlink(missing_ok=True)

# Rebuild from Electron's verified original on every refresh, including when a
# prior staging tree still uses the old launcher name.
with tempfile.TemporaryDirectory(dir=release) as temp:
    tmp = Path(temp)
    stock = release / 'electron.exe'
    if refresh_only:
        with zipfile.ZipFile(work / 'electron.zip') as archive:
            stock = Path(archive.extract('electron.exe', tmp))
    output = tmp / launcher.name
    subprocess.run(['node', str(work / 'asar-tool/brand-launcher.mjs'), str(stock),
                    str(output), product_name,
                    upstream_manifest['version'], str(app / 'resources/icon.ico')], check=True)
    output.replace(launcher)
    stock.unlink()
old_launcher.unlink(missing_ok=True)
if container.exists():
    shutil.rmtree(container)
container.mkdir()
for name in ['dist', 'dist-electron']:
    shutil.copytree(app/name, container/name, ignore=shutil.ignore_patterns('*.map'))
for name in ['LICENSE.txt', 'NOTICES.md']:
    shutil.copy2(app/name, container/name)
manifest = {key: upstream_manifest[key] for key in ('version', 'type', 'main', 'license', 'homepage')}
manifest.update(name='ltx-desktop-amd', productName=product_name, description='Community branch of LTX Desktop for Windows NVIDIA and AMD GPUs')
(container/'package.json').write_text(json.dumps(manifest,indent=2)+'\n')

# Vite bundles the renderer dependencies into the UI. The main process uses
# Koffi through a runtime require; ship it plus its Windows N-API addon.
# Include the Electron-side packages upstream uses even if Vite inlines them.
installed = {}
def add_package(src, dest_parent):
    src = src.resolve(strict=True)
    info = json.loads((src/'package.json').read_text())
    name = info['name']
    dest = container/'node_modules'/name
    if dest in installed and installed[dest] != src:
        dest = dest_parent/'node_modules'/name
    if dest in installed:
        if installed[dest] != src:
            raise RuntimeError(f'unresolved version collision: {name}')
        return
    installed[dest] = src
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns('node_modules', '*.map'))
    for dep in info.get('dependencies',{}):
        node_modules = next(parent for parent in src.parents if parent.name == 'node_modules')
        add_package(node_modules/dep, dest)
for name in ['koffi','electron-updater','js-yaml','zod']:
    add_package(app/'node_modules'/name, container)
with tarfile.open(work/'koffi-win32-x64.tgz','r:gz') as archive:
    pkg = container/'node_modules/@koromix/koffi-win32-x64'
    for member in archive:
        if member.isfile():
            target = pkg / Path(member.name).relative_to('package')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as payload, target.open('wb') as dest:
                shutil.copyfileobj(payload, dest)
koffi = resources/'koffi/win32_x64/koffi.node'
koffi.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(pkg/'win32_x64/koffi.node',koffi)
asar_cli = work/'asar-tool/node_modules/@electron/asar/bin/asar.mjs'
archive = resources/'app.asar'
subprocess.run(['node',str(asar_cli),'pack',str(container),str(archive),'--unpack','{**/koffi.node,**/dist/splash/**}'],check=True)
listing = subprocess.check_output(['node',str(asar_cli),'list',str(archive)],text=True).splitlines()
for required in ['dist/index.html','dist-electron/main.js','dist-electron/preload.js','package.json','node_modules/koffi/index.cjs']:
    if '/'+required not in listing:
        raise RuntimeError('missing ASAR entry: '+required)
for src,dest in [(app/'resources/explore-assets',resources/'explore-assets'),(app/'dist-remote',resources/'dist-remote')]:
    shutil.copytree(src,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.map'))
shutil.copy2(app/'resources/icon.ico', resources/'icon.ico')
for path in (app/'backend').rglob('*'):
    if path.is_file() and not {'tests','__pycache__','.venv'}.intersection(path.parts) and (path.suffix in ['.py','.sql','.json'] or path.name=='pyproject.toml'):
        dest = resources/path.relative_to(app)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
(resources/'shared').mkdir(exist_ok=True)
for path in (app/'shared').glob('*.json'):
    shutil.copy2(path,resources/'shared'/path.name)
shutil.copytree(app / 'resources/licenses/minimax-h3', resources / 'licenses/minimax-h3', dirs_exist_ok=True)
engine_resources = resources / 'comfy-engine'
engine_resources.mkdir(exist_ok=True)
for filename, expected in {
    'comfyui-v0.38.0.zip': '4a144a553c77397c81f6067f3f8c8ab9dffd42889af48fb35fec2c64a71c75b2',
    'kandinsky6-1.0.2.zip': '5ddf0bef14d0e571f8760403d270f2b1d6f9b6b10e56b8b707743c15751b7f80',
}.items():
    source = app.parent / 'vendor/ComfyUI' / filename
    verify(source, 'sha256', bytes.fromhex(expected))
    shutil.copy2(source, engine_resources / filename)
subprocess.run([
    sys.executable, str(app.parent / 'runtimework/prepare_comfy_deps.py'),
    '--out', str(engine_resources / 'site-packages'),
    '--cache', str(app.parent / 'build/.cache/comfy-wheels'),
], check=True)
print(f'{release}: {len(listing)} ASAR entries; native node addon {koffi.stat().st_size:,} bytes',flush=True)

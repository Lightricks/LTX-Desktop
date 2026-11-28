"""Scan FastAPI route files and generate an HTML API endpoint reference.

Usage:
    python generate_api_docs.py

Outputs: generated/api_docs_<commit-sha>_<timestamp>.html
"""

from __future__ import annotations

import html
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROUTES_DIR = SCRIPT_DIR / "_routes"
API_TYPES_FILE = SCRIPT_DIR / "api_types.py"
# Extra source for types not defined in api_types.py
EXTRA_TYPE_SOURCES: list[Path] = [
    SCRIPT_DIR / "state" / "app_settings.py",
]
GENERATED_DIR = SCRIPT_DIR / "generated"

# ---------------------------------------------------------------------------
# Regex patterns for FastAPI conventions
# ---------------------------------------------------------------------------
# Matches: router = APIRouter(prefix="/api/ic-lora", ...)
RE_ROUTER_PREFIX = re.compile(
    r"""APIRouter\([^)]*prefix\s*=\s*["']([^"']+)["']""",
)
# Matches: @router.get("/path", response_model=SomeModel)
#      or  @router.post("/path", response_model=SomeModel)
RE_ROUTE_DECORATOR = re.compile(
    r"""@router\.(get|post|put|patch|delete)\(\s*["']([^"']+)["']"""
    r"""(?:.*?response_model\s*=\s*([\w\[\], ]+))?""",
    re.DOTALL,
)
# Matches the def line right after the decorator (possibly multi-line decorator)
RE_DEF_AFTER_DECORATOR = re.compile(
    r"""@router\.\w+\([^)]*\)\s*\ndef\s+(\w+)\s*\(([^)]*)\)""",
    re.DOTALL,
)
# Matches a typed parameter like `req: GenerateVideoRequest`
# Must NOT match Depends(...), Form(...), File(...), or bare handler params
RE_REQUEST_PARAM = re.compile(
    r"""(\w+)\s*:\s*([\w.]+)\s*(?:$|,)""",
)

# ---------------------------------------------------------------------------
# Type definition extraction from source files
# ---------------------------------------------------------------------------
TYPE_ALIAS_RE = re.compile(
    r"""^(\w+)\s*(?::\s*TypeAlias\s*)?\s*=\s*(\w+)""", re.MULTILINE
)


def extract_class_definitions(source: str) -> dict[str, str]:
    """Return {ClassName: full_class_source} for every class in *source*."""
    classes: dict[str, str] = {}
    lines = source.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        m = re.match(r"^class\s+(\w+)\s*(?:\([^)]*\))?\s*:", lines[i])
        if m:
            name = m.group(1)
            start = i
            i += 1
            # Collect the class body (indented lines, blank lines within body)
            while i < len(lines):
                line = lines[i]
                # A non-blank, non-indented line ends the class
                if line.strip() and not line[0].isspace():
                    break
                i += 1
            classes[name] = "".join(lines[start:i]).rstrip("\n")
        else:
            i += 1
    return classes


def extract_type_aliases(source: str) -> dict[str, str]:
    """Return {AliasName: TargetName} for simple type aliases."""
    aliases: dict[str, str] = {}
    for m in TYPE_ALIAS_RE.finditer(source):
        alias_name, target = m.group(1), m.group(2)
        # Skip private names, imports, and common non-alias assignments
        if alias_name.startswith("_") or alias_name[0].islower():
            continue
        aliases[alias_name] = target
    return aliases


def load_type_definitions() -> dict[str, str]:
    """Load class definitions from api_types.py and extra sources."""
    all_defs: dict[str, str] = {}
    all_aliases: dict[str, str] = {}
    sources = [API_TYPES_FILE, *EXTRA_TYPE_SOURCES]
    for path in sources:
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        all_defs.update(extract_class_definitions(src))
        all_aliases.update(extract_type_aliases(src))
    # Resolve aliases: e.g. SettingsResponse -> AppSettings
    for alias, target in all_aliases.items():
        if target in all_defs and alias not in all_defs:
            all_defs[alias] = f"# {alias} = {target}\n\n{all_defs[target]}"
    return all_defs


# ---------------------------------------------------------------------------
# Route scanning
# ---------------------------------------------------------------------------

# Types to ignore when looking for "request body" parameters
SKIP_TYPES = {
    "AppHandler",
    "UploadFile",
    "str",
    "int",
    "float",
    "bool",
    "None",
    "list",
    "dict",
}


def scan_routes() -> list[dict[str, str]]:
    """Return a list of endpoint dicts from the _routes directory."""
    endpoints: list[dict[str, str]] = []

    for pyfile in sorted(ROUTES_DIR.glob("*.py")):
        if pyfile.name.startswith("_"):
            continue
        source = pyfile.read_text(encoding="utf-8")

        # 1. Determine the router prefix (may be empty)
        prefix_m = RE_ROUTER_PREFIX.search(source)
        prefix = prefix_m.group(1) if prefix_m else ""

        # 2. Find all decorator + def pairs
        # We use a combined regex that captures the decorator and the def together.
        pattern = re.compile(
            r"""@router\.(get|post|put|patch|delete)\(\s*["']([^"']+)["']"""
            r"""([^)]*)\)\s*\n"""
            r"""def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([\w\[\], |]+))?\s*:""",
            re.DOTALL,
        )

        for m in pattern.finditer(source):
            method = m.group(1).upper()
            path = prefix + m.group(2)
            decorator_args = m.group(3)
            func_name = m.group(4)
            params_str = m.group(5)
            return_annotation = m.group(6) or ""

            # Extract response_model from decorator args
            resp_model_m = re.search(
                r"response_model\s*=\s*([\w\[\], ]+)", decorator_args
            )
            response_type = (
                resp_model_m.group(1).strip() if resp_model_m else return_annotation.strip()
            )

            # Extract request type from function params
            request_type = ""
            if params_str:
                for param_m in re.finditer(
                    r"(\w+)\s*:\s*([\w.]+)", params_str
                ):
                    pname = param_m.group(1)
                    ptype = param_m.group(2)
                    # Skip dependency-injected params and primitives
                    if ptype in SKIP_TYPES or pname == "handler":
                        continue
                    # Skip Form/File params (detected by looking ahead for = Form/= File)
                    after = params_str[param_m.end():]
                    if re.match(r"\s*=\s*(Form|File|Depends)\s*\(", after):
                        continue
                    request_type = ptype
                    break  # Take first non-DI param

            # Clean up: strip "list[ModelInfo]" style to keep it readable
            endpoints.append(
                {
                    "file": pyfile.name,
                    "name": func_name,
                    "method": method,
                    "path": path,
                    "request_type": request_type,
                    "response_type": response_type,
                }
            )

    return endpoints


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

# Lower index = higher priority when methods share the same path
METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}


def sort_endpoints(endpoints: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort by: file name, then GET before POST, then path alphabetically."""
    return sorted(
        endpoints,
        key=lambda ep: (
            ep["file"],
            METHOD_ORDER.get(ep["method"], 9),
            ep["path"],
        ),
    )


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def get_git_short_sha() -> str:
    """Return the short SHA of HEAD, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_output_path() -> Path:
    sha = get_git_short_sha()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return GENERATED_DIR / f"api_docs_{sha}_{ts}.html"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "GET": "#61affe",
    "POST": "#49cc90",
    "PUT": "#fca130",
    "PATCH": "#50e3c2",
    "DELETE": "#f93e3e",
}


def type_cell(type_name: str, type_defs: dict[str, str]) -> str:
    """Render a table cell for a type, with a clickable link if definition exists."""
    if not type_name:
        return '<td class="type-cell">—</td>'

    # Handle wrapper types like list[ModelInfo]
    inner_types = re.findall(r"\b([A-Z]\w+)\b", type_name)
    display = html.escape(type_name)

    linkable = [t for t in inner_types if t in type_defs]
    if linkable:
        for t in linkable:
            display = display.replace(
                t,
                f'<a href="#" class="type-link" data-type="{html.escape(t)}">{t}</a>',
            )
        return f'<td class="type-cell">{display}</td>'
    return f'<td class="type-cell">{display}</td>'


def build_html(
    endpoints: list[dict[str, str]], type_defs: dict[str, str]
) -> str:
    """Build the complete HTML document."""

    # Prepare type definition JSON for embedding
    type_json_entries: list[str] = []
    for name, body in sorted(type_defs.items()):
        escaped = body.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        type_json_entries.append(f'  "{html.escape(name)}": `{escaped}`')
    type_json = "{\n" + ",\n".join(type_json_entries) + "\n}"

    # Build table rows
    rows: list[str] = []
    for ep in endpoints:
        color = METHOD_COLORS.get(ep["method"], "#888")
        rows.append(
            f"<tr>"
            f'<td class="route-name"><code>{html.escape(ep["name"])}</code></td>'
            f'<td><span class="method-badge" style="background:{color}">'
            f'{ep["method"]}</span></td>'
            f'<td class="path-cell"><code>{html.escape(ep["path"])}</code></td>'
            f'{type_cell(ep["request_type"], type_defs)}'
            f'{type_cell(ep["response_type"], type_defs)}'
            f'<td class="file-cell">{html.escape(ep["file"])}</td>'
            f"</tr>"
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LTX Desktop - AMD 395 MAX — API Endpoints</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --row-hover: #1c2129;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 1.75rem;
    margin-bottom: .25rem;
  }}
  .subtitle {{
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    font-size: .9rem;
  }}
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: .875rem;
  }}
  thead th {{
    background: var(--surface);
    text-align: left;
    padding: .75rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: .75rem;
    letter-spacing: .05em;
    position: sticky;
    top: 0;
    z-index: 2;
  }}
  tbody td {{
    padding: .6rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--row-hover); }}
  .method-badge {{
    display: inline-block;
    padding: .15rem .55rem;
    border-radius: 4px;
    color: #fff;
    font-weight: 700;
    font-size: .75rem;
    letter-spacing: .03em;
    min-width: 3.2rem;
    text-align: center;
  }}
  .route-name code {{
    color: var(--text-muted);
    font-size: .8rem;
  }}
  .path-cell code {{
    color: var(--accent);
    font-weight: 500;
  }}
  .file-cell {{
    color: var(--text-muted);
    font-size: .8rem;
  }}
  .type-cell {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: .82rem; }}
  .type-link {{
    color: var(--accent);
    text-decoration: none;
    cursor: pointer;
    border-bottom: 1px dashed var(--accent);
  }}
  .type-link:hover {{ color: #79c0ff; border-bottom-style: solid; }}

  /* Modal overlay */
  .modal-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.6);
    backdrop-filter: blur(4px);
    z-index: 100;
    justify-content: center;
    align-items: center;
  }}
  .modal-overlay.active {{ display: flex; }}
  .modal {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    width: min(640px, 90vw);
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 16px 48px rgba(0,0,0,.4);
  }}
  .modal-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: .75rem 1.25rem;
    border-bottom: 1px solid var(--border);
  }}
  .modal-header h3 {{
    font-size: 1rem;
    color: var(--accent);
  }}
  .modal-close {{
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1.4rem;
    cursor: pointer;
    padding: .25rem;
    line-height: 1;
  }}
  .modal-close:hover {{ color: var(--text); }}
  .modal-body {{
    padding: 1rem 1.25rem;
    overflow-y: auto;
  }}
  .modal-body pre {{
    margin: 0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: .82rem;
    line-height: 1.65;
    white-space: pre;
    overflow-x: auto;
  }}

  /* Syntax highlighting */
  .kw {{ color: #ff7b72; }}
  .cls {{ color: #f0883e; }}
  .tp {{ color: #79c0ff; }}
  .str {{ color: #a5d6ff; }}
  .num {{ color: #79c0ff; }}
  .cmt {{ color: #8b949e; font-style: italic; }}
  .op {{ color: #ff7b72; }}
  .fn {{ color: #d2a8ff; }}
  .dec {{ color: #d2a8ff; }}
  .const {{ color: #79c0ff; }}

  .count-badge {{
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .1rem .6rem;
    font-size: .8rem;
    color: var(--text-muted);
    margin-left: .5rem;
    vertical-align: middle;
  }}
</style>
</head>
<body>

<h1>LTX Desktop - AMD 395 MAX — API Endpoints <span class="count-badge">{len(endpoints)}</span></h1>
<p class="subtitle">Auto-generated from <code>_routes/</code> — click any type name to inspect its definition</p>

<div class="table-wrap">
<table>
<thead>
<tr>
  <th>Route Name</th>
  <th>Method</th>
  <th>Path</th>
  <th>Request Type</th>
  <th>Response Type</th>
  <th>File</th>
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</div>

<!-- Type definition modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modal-title"></h3>
      <button class="modal-close" id="modal-close">&times;</button>
    </div>
    <div class="modal-body">
      <pre id="modal-code"></pre>
    </div>
  </div>
</div>

<script>
// Type definitions embedded as template literals
const TYPE_DEFS = {type_json};

// Minimal Python syntax highlighter
function highlightPython(code) {{
  // Escape HTML first
  let s = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  // Order matters — do comments/strings first to avoid clashing
  // Strings (triple-quoted, then single/double)
  s = s.replace(/(\"\"\"[\\s\\S]*?\"\"\"|\'\'\'[\\s\\S]*?\'\'\')/g, '<span class="str">$1</span>');
  s = s.replace(/(\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\"|\'[^\'\\\\]*(?:\\\\.[^\'\\\\]*)*\')/g, '<span class="str">$1</span>');
  // Comments
  s = s.replace(/(#.*)/gm, '<span class="cmt">$1</span>');
  // Decorators
  s = s.replace(/^(@\\w+)/gm, '<span class="dec">$1</span>');
  // Keywords
  const kws = 'class|def|return|if|else|elif|import|from|as|None|True|False|and|or|not|in|is|for|while|with|try|except|raise|pass|yield|lambda|async|await'.split('|');
  const kwRe = new RegExp('\\\\b(' + kws.join('|') + ')\\\\b', 'g');
  s = s.replace(kwRe, '<span class="kw">$1</span>');
  // Type names (PascalCase)
  s = s.replace(/\\b([A-Z][a-zA-Z0-9]+)\\b/g, '<span class="tp">$1</span>');
  // Numbers
  s = s.replace(/\\b(\\d+(?:\\.\\d+)?)\\b/g, '<span class="num">$1</span>');
  // Operators
  s = s.replace(/(=&gt;|->|\\|)/g, '<span class="op">$1</span>');

  return s;
}}

// Modal logic
const overlay = document.getElementById('modal-overlay');
const modalTitle = document.getElementById('modal-title');
const modalCode = document.getElementById('modal-code');
const closeBtn = document.getElementById('modal-close');

document.addEventListener('click', (e) => {{
  const link = e.target.closest('.type-link');
  if (!link) return;
  e.preventDefault();
  const typeName = link.dataset.type;
  const def = TYPE_DEFS[typeName];
  if (!def) return;
  modalTitle.textContent = typeName;
  modalCode.innerHTML = highlightPython(def);
  overlay.classList.add('active');
}});

closeBtn.addEventListener('click', () => overlay.classList.remove('active'));
overlay.addEventListener('click', (e) => {{
  if (e.target === overlay) overlay.classList.remove('active');
}});
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') overlay.classList.remove('active');
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    type_defs = load_type_definitions()
    endpoints = sort_endpoints(scan_routes())

    output_path = make_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = build_html(endpoints, type_defs)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Generated {output_path}  ({len(endpoints)} endpoints)")


if __name__ == "__main__":
    main()

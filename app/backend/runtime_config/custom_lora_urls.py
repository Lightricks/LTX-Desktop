"""Rules for the URL a user pastes to add a custom LoRA.

Two kinds of link are supported: a Hugging Face model page and a Hugging Face file link.
Civitai and other direct HTTPS links to a ``.safetensors`` file are also parsed below, but
they are not supported and not tested. Everything else fails before a request leaves the
machine. The same rules run again on every redirect target, because a download link often
redirects to a CDN host.

A Hugging Face model page names no file. ``resolve_custom_lora_url`` returns the repo, the
handler lists its files, and ``pick_hugging_face_weights`` chooses the one ``.safetensors`` file.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, unquote, urlsplit

# The in-app Hugging Face token goes to these hosts only. A redirect target (a signed CDN URL)
# never gets it.
HF_HOSTS = frozenset({"huggingface.co", "hf.co"})
CIVITAI_HOSTS = frozenset({"civitai.com", "www.civitai.com"})

_CIVITAI_DOWNLOAD_PATH = re.compile(r"^/api/download/models/(\d+)/?$")
_CIVITAI_PAGE_PATH = re.compile(r"^/models/\d+")
# A model page is /<owner>/<repo>, optionally with /tree/<rev>. A file page is
# /<owner>/<repo>/blob/<rev>/<file>. The file page differs from the download link by one word.
# Owner and name use the characters Hugging Face allows, so they are safe in an API path.
_HF_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_HF_MODEL_PAGE_PATH = re.compile(
    rf"^/(?P<owner>{_HF_NAME})/(?P<name>{_HF_NAME})(?:/tree/(?P<revision>[^/]+))?/?$"
)
# Any link to a repo: model page, /tree/, /blob/ or /resolve/ file link. The revision is optional.
_HF_REPO_LINK_PATH = re.compile(
    rf"^/(?P<owner>{_HF_NAME})/(?P<name>{_HF_NAME})(?:/(?:tree|blob|resolve)/(?P<revision>[^/]+))?(?:/|$)"
)
_HF_FILE_PATH = re.compile(
    rf"^/(?P<owner>{_HF_NAME})/(?P<name>{_HF_NAME})/(?:blob|resolve)/(?P<revision>[^/]+)/(?P<file>.+\.safetensors)$",
    re.IGNORECASE,
)
_HF_OWNER_PATH = re.compile(rf"^/(?P<owner>{_HF_NAME})/(?P<name>{_HF_NAME})(?:/|$)")
# Top-level Hugging Face sections that look like /<owner>/<repo> but are not model repos.
_HF_RESERVED_OWNERS = frozenset(
    {"api", "blog", "collections", "datasets", "docs", "models", "papers", "settings", "spaces"}
)
_HF_DEFAULT_REVISION = "main"
_MAX_LISTED_NAMES = 4


class CustomLoraUrlError(ValueError):
    """The URL is not accepted. The message is safe to show to the user."""


@dataclass(frozen=True, slots=True)
class HuggingFaceRepo:
    """A Hugging Face model repository named by a model-page link."""

    repo_id: str
    revision: str = _HF_DEFAULT_REVISION

    @property
    def page_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    def file_url(self, filename: str) -> str:
        path = "/".join(quote(part, safe="") for part in filename.split("/"))
        return f"{self.page_url}/resolve/{quote(self.revision, safe='')}/{path}"


@dataclass(frozen=True, slots=True)
class CustomLoraUrl:
    """A checked URL. ``repo`` is set when the link is a model page: it still names no file."""

    url: str
    host: str
    repo: HuggingFaceRepo | None = None

    @property
    def is_hugging_face(self) -> bool:
        return self.host in HF_HOSTS


_TOP_LEVEL_LABEL = re.compile(r"[a-z][a-z0-9-]*")


def _host_is_local(host: str) -> bool:
    """Whether the host is, or can be written as, this computer or a private network.

    The check is on the text of the host, before any DNS lookup. The downloader checks the
    resolved addresses again at connect time (``public_address_adapter``), which catches a public
    name that points to a private address.
    """
    # "localhost." is the same host as "localhost": the trailing dot is the DNS root.
    host = host.rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return not address.is_global
    # Other numeric spellings (127.1, 2130706433, 0x7f.1) are not IP literals for ipaddress, but
    # the system resolver reads them as one. A real host name has a letter in its last label and
    # a dot, so a single label such as "intranet" (a private name) is refused too.
    labels = host.split(".")
    return len(labels) < 2 or _TOP_LEVEL_LABEL.fullmatch(labels[-1]) is None


def check_fetch_url(url: str) -> CustomLoraUrl:
    """Check one URL for a request: HTTPS, a real host, no private network address.

    Used for the pasted URL (after ``resolve_custom_lora_url``) and for each redirect target.
    """
    parts = urlsplit(url.strip())
    if parts.scheme != "https":
        raise CustomLoraUrlError("Use an https:// link.")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host or parts.username is not None or parts.password is not None:
        raise CustomLoraUrlError("The link is not valid.")
    if _host_is_local(host):
        raise CustomLoraUrlError("The link points to this computer or a private network.")
    return CustomLoraUrl(url=parts.geturl(), host=host)


def _decoded_path(path: str) -> str:
    try:
        return unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise CustomLoraUrlError("The link is not valid.") from exc


def resolve_custom_lora_url(url: str) -> CustomLoraUrl:
    """Turn a pasted link into the file URL to fetch, or raise ``CustomLoraUrlError``.

    Only Hugging Face model pages and file links are supported sources. The Civitai and direct
    ``.safetensors`` branches below are not a supported source (most Civitai files require a
    sign-in).
    """
    checked = check_fetch_url(url)
    parts = urlsplit(checked.url)
    decoded_path = _decoded_path(parts.path)
    if checked.host in HF_HOSTS:
        return _resolve_hugging_face(checked, decoded_path)
    if checked.host not in CIVITAI_HOSTS:
        if not decoded_path.lower().endswith(".safetensors"):
            raise CustomLoraUrlError("Use a direct link to a .safetensors file.")
        return checked
    if _CIVITAI_DOWNLOAD_PATH.match(parts.path):
        return checked
    if _CIVITAI_PAGE_PATH.match(parts.path):
        versions = parse_qs(parts.query).get("modelVersionId", [])
        if len(versions) == 1 and versions[0].isdigit():
            return check_fetch_url(f"https://civitai.com/api/download/models/{versions[0]}")
        raise CustomLoraUrlError(
            "This is a Civitai page. Open the version you want, then copy the Download link "
            "(civitai.com/api/download/models/...)."
        )
    raise CustomLoraUrlError(
        "Use a Civitai download link (civitai.com/api/download/models/...)."
    )


def _resolve_hugging_face(checked: CustomLoraUrl, decoded_path: str) -> CustomLoraUrl:
    """A file page (``/blob/``) becomes its download link. A model page returns its repo."""
    parts = urlsplit(checked.url)
    file_link = _HF_FILE_PATH.match(parts.path)
    if (
        file_link
        and file_link.group("owner") not in _HF_RESERVED_OWNERS
        and decoded_path.lower().endswith(".safetensors")
    ):
        owner = file_link.group("owner")
        name = file_link.group("name")
        revision = file_link.group("revision")
        file_part = file_link.group("file")
        resolved_url = parts._replace(path=f"/{owner}/{name}/resolve/{revision}/{file_part}").geturl()
        return check_fetch_url(resolved_url)
    model_page = _HF_MODEL_PAGE_PATH.match(parts.path)
    if model_page and model_page.group("owner") not in _HF_RESERVED_OWNERS:
        repo = HuggingFaceRepo(
            repo_id=f"{model_page.group('owner')}/{model_page.group('name')}",
            revision=model_page.group("revision") or _HF_DEFAULT_REVISION,
        )
        return CustomLoraUrl(url=checked.url, host=checked.host, repo=repo)
    raise CustomLoraUrlError("Use a Hugging Face model page or a .safetensors file link.")


def hugging_face_repo_of(url: str) -> HuggingFaceRepo | None:
    """The repo a Hugging Face link points to, or ``None`` for any other link.

    The model page lists the README (the model card). This accepts any link kind that
    names a repo. It does not raise: a link without a card has nothing to suggest.
    """
    parts = urlsplit(url.strip())
    if parts.scheme != "https" or (parts.hostname or "").lower() not in HF_HOSTS:
        return None
    match = _HF_REPO_LINK_PATH.match(parts.path)
    if match is None or match.group("owner") in _HF_RESERVED_OWNERS:
        return None
    return HuggingFaceRepo(
        repo_id=f"{match.group('owner')}/{match.group('name')}",
        revision=match.group("revision") or _HF_DEFAULT_REVISION,
    )


def pick_hugging_face_weights(filenames: Sequence[str]) -> str:
    """The one ``.safetensors`` file of a repo, or ``CustomLoraUrlError`` with what to do next.

    A repo has other files (the README, ``.gitattributes``) that are not weights. Two or more
    weights files are ambiguous: a guess could load the wrong LoRA, so the user picks the file.
    """
    weights = sorted(name for name in filenames if name.lower().endswith(".safetensors"))
    if not weights:
        raise CustomLoraUrlError(
            "This Hugging Face repository has no .safetensors file. Use a repository "
            "that contains a LoRA file."
        )
    if len(weights) > 1:
        shown = ", ".join(weights[:_MAX_LISTED_NAMES])
        more = "" if len(weights) <= _MAX_LISTED_NAMES else f" and {len(weights) - _MAX_LISTED_NAMES} more"
        raise CustomLoraUrlError(
            f"This Hugging Face repository has more than one .safetensors file ({shown}{more}). "
            "Open the file you want and copy its link."
        )
    return weights[0]


def page_url_for(fetch_url: str) -> str | None:
    """The Hugging Face model page of a fetch URL, or ``None`` for any other host.

    The details screen links to this page only. A file the user chose from disk has no page.
    """
    parts = urlsplit(fetch_url)
    if (parts.hostname or "").lower() not in HF_HOSTS:
        return None
    match = _HF_OWNER_PATH.match(parts.path)
    if match is None or match.group("owner") in _HF_RESERVED_OWNERS:
        return None
    return HuggingFaceRepo(repo_id=f"{match.group('owner')}/{match.group('name')}").page_url


def saved_source_url(fetch_url: str) -> str:
    """The link to keep in ``meta.json`` so the weights can be fetched again.

    CWE-312: a pasted link can carry a secret in its query string (for example a ``?token=``
    query or a signed CDN URL). The query and fragment are dropped for every host before the
    link is saved on disk. The file is found by its path. A link that needs its query to work
    fails when the weights are fetched again, and the user then adds the file.
    """
    return urlsplit(fetch_url)._replace(query="", fragment="").geturl()

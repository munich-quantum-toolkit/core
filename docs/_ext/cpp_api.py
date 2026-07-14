# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Cross-reference native Doxygen HTML from Sphinx."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
from xml.etree import ElementTree as ET  # noqa: S405

from docutils import nodes
from sphinx.domains import Domain
from sphinx.errors import ExtensionError
from sphinx.roles import XRefRole
from sphinx.util.osutil import relative_uri
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sphinx import addnodes
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment
    from sphinx.util.typing import ExtensionMetadata


@dataclass(frozen=True)
class ApiTarget:
    """A link target read from a Doxygen tag file."""

    kind: str
    uri: str


class ApiXRefRole(XRefRole):
    """Cross-reference role supporting Sphinx's shortened-title syntax."""

    @override
    def process_link(
        self,
        env: BuildEnvironment,
        refnode: addnodes.pending_xref,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        """Strip a leading tilde and shorten the displayed qualified name.

        Returns:
            The display title and normalized reference target.
        """
        title, target = super().process_link(env, refnode, has_explicit_title, title, target)
        if not has_explicit_title and target.startswith("~"):
            target = target[1:]
            title = title.lstrip("~").rsplit("::", maxsplit=1)[-1]
        return title, target


def read_tagfile(path: Path, base_uri: str) -> dict[str, ApiTarget]:
    """Read symbol names and HTML targets from a Doxygen tag file.

    Returns:
        The API targets indexed by their qualified and unqualified names.
    """
    root = ET.parse(path).getroot()  # noqa: S314
    targets: dict[str, ApiTarget] = {}

    for compound in root.findall("compound"):
        compound_kind = compound.get("kind", "object")
        compound_name = compound.findtext("name", "")
        compound_file = compound.findtext("filename", "")
        if compound_name and compound_file:
            targets.setdefault(compound_name, ApiTarget(compound_kind, urljoin(base_uri, compound_file)))

        for member in compound.findall("member"):
            member_kind = member.get("kind", "object")
            member_name = member.findtext("name", "")
            anchor_file = member.findtext("anchorfile", "")
            anchor = member.findtext("anchor", "")
            if not member_name or not anchor_file:
                continue

            uri = urljoin(base_uri, anchor_file)
            if anchor:
                uri = f"{uri}#{anchor}"
            target = ApiTarget(member_kind, uri)
            targets.setdefault(member_name, target)

            if compound_kind in {"class", "concept", "namespace", "struct", "union"}:
                qualified_name = f"{compound_name}::{member_name}"
                targets.setdefault(qualified_name, target)
            else:
                qualified_name = member_name

            arglist = member.findtext("arglist", "")
            if arglist:
                targets.setdefault(f"{qualified_name}{arglist}", target)

    return targets


def read_xml_inventory(path: Path, base_uri: str) -> dict[str, ApiTarget]:
    """Read symbols omitted from tag files from Doxygen's XML output.

    Returns:
        The API targets indexed by their qualified and unqualified names.
    """
    targets: dict[str, ApiTarget] = {}
    index = ET.parse(path / "index.xml").getroot()  # noqa: S314
    member_targets: dict[str, ApiTarget] = {}

    for compound in index.findall("compound"):
        refid = compound.get("refid", "")
        kind = compound.get("kind", "object")
        name = compound.findtext("name", "")
        html_file = path.parent / "html" / f"{refid}.html"
        if not refid:
            continue
        if name and html_file.is_file():
            targets.setdefault(name, ApiTarget(kind, urljoin(base_uri, html_file.name)))

        for member in compound.findall("member"):
            member_refid = member.get("refid", "")
            member_kind = member.get("kind", "object")
            member_name = member.findtext("name", "")
            if not member_refid or not member_name:
                continue
            member_html = path.parent / "html" / f"{member_refid.rsplit('_1', maxsplit=1)[0]}.html"
            if not member_html.is_file():
                continue
            anchor = member_refid.rsplit("_1", maxsplit=1)[-1]
            target = ApiTarget(member_kind, f"{urljoin(base_uri, member_html.name)}#{anchor}")
            member_targets[member_refid] = target
            targets.setdefault(member_name, target)

        compound_xml = path / f"{refid}.xml"
        if not compound_xml.is_file():
            continue
        root = ET.parse(compound_xml).getroot()  # noqa: S314
        for member in root.findall(".//memberdef"):
            member_refid = member.get("id", "")
            member_name = member.findtext("name", "")
            qualified_name = member.findtext("qualifiedname", member_name)
            target = member_targets.get(member_refid)
            if target is None or not qualified_name:
                continue
            targets.setdefault(member_name, target)
            targets.setdefault(qualified_name, target)

            args = member.findtext("argsstring", "")
            if args:
                targets.setdefault(f"{qualified_name}{args}", target)

    return targets


class TagfileDomain(Domain):
    """Base domain resolving references through a Doxygen tag file."""

    roles: ClassVar[dict[str, XRefRole]] = {
        role: ApiXRefRole() for role in ("any", "class", "concept", "enum", "func", "struct", "type", "var")
    }
    directives: ClassVar[dict[str, object]] = {}
    initial_data: ClassVar[dict[str, object]] = {}
    config_name: ClassVar[str]

    def _targets(self) -> dict[str, ApiTarget]:
        cached = self.data.get("targets")
        if isinstance(cached, dict):
            return cached

        config = getattr(self.env.config, self.config_name)
        tagfile, base_uri = config[:2]
        path = Path(tagfile)
        if not path.is_absolute():
            path = Path(self.env.srcdir) / path
        targets = read_tagfile(path, base_uri)
        if len(config) == 3:
            xml_path = Path(config[2])
            if not xml_path.is_absolute():
                xml_path = Path(self.env.srcdir) / xml_path
            targets.update(read_xml_inventory(xml_path, base_uri))
        self.data["targets"] = targets
        return targets

    @override
    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        typ: str,
        target: str,
        node: nodes.Element,
        contnode: nodes.Element,
    ) -> nodes.reference | None:
        """Resolve a symbol reference to native Doxygen HTML.

        Returns:
            A link to the Doxygen target, or ``None`` if it is unknown.
        """
        del env, typ, node
        api_target = self._targets().get(target)
        if api_target is None:
            return None

        parsed = urlparse(api_target.uri)
        if parsed.scheme or parsed.netloc:
            refuri = api_target.uri
            internal = False
        else:
            refuri = relative_uri(builder.get_target_uri(fromdocname), parsed.path)
            if parsed.fragment:
                refuri = f"{refuri}#{parsed.fragment}"
            internal = True
        return nodes.reference("", "", contnode, internal=internal, refuri=refuri)

    @override
    def get_objects(self) -> Iterable[tuple[str, str, str, str, str, int]]:
        """Return no local Sphinx objects; all targets live in Doxygen HTML."""
        return ()

    @override
    def resolve_any_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        target: str,
        node: nodes.Element,
        contnode: nodes.Element,
    ) -> list[tuple[str, nodes.reference]]:
        """Resolve an ``any`` reference through the tag-file inventory.

        Returns:
            The matching domain role and link, or an empty list if unknown.
        """
        resolved = self.resolve_xref(env, fromdocname, builder, "any", target, node, contnode)
        if resolved is None:
            return []
        api_target = self._targets()[target]
        return [(f"{self.name}:{api_target.kind}", resolved)]


class CppApiDomain(TagfileDomain):
    """Domain for the MQT Core C++ API."""

    name = "cpp-api"
    label = "MQT Core C++ API"
    config_name = "cpp_api_tagfile"


class QdmiApiDomain(TagfileDomain):
    """Domain for the versioned external QDMI API."""

    name = "qdmi"
    label = "QDMI API"
    config_name = "qdmi_api_tagfile"


def _path_from_config(app: Sphinx, path: str) -> Path:
    """Resolve a documentation-relative path from an extension setting.

    Returns:
        The absolute path represented by ``path``.
    """
    configured_path = Path(path)
    return configured_path if configured_path.is_absolute() else Path(app.srcdir) / configured_path


def build_doxygen(app: Sphinx) -> None:
    """Download the QDMI inventory and generate the native C++ API.

    Raises:
        ExtensionError: If the QDMI inventory cannot be downloaded or Doxygen
            cannot generate the API documentation.
    """
    tagfile = _path_from_config(app, app.config.qdmi_api_tagfile[0])
    tagfile.parent.mkdir(parents=True, exist_ok=True)
    doxygen = shutil.which("doxygen")
    if doxygen is None:
        msg = "Doxygen is required to build the native C++ API documentation"
        raise ExtensionError(msg)
    try:
        with urlopen(app.config.qdmi_api_tagfile_url, timeout=30) as response:  # noqa: S310
            tagfile.write_bytes(response.read())
        subprocess.run([doxygen, "Doxyfile"], cwd=app.srcdir, check=True)  # noqa: S603
    except (OSError, subprocess.CalledProcessError) as error:
        msg = "Unable to generate the native C++ API documentation"
        raise ExtensionError(msg) from error


def publish_doxygen_html(app: Sphinx, exception: Exception | None) -> None:
    """Publish native Doxygen HTML alongside an HTML Sphinx build."""
    if exception is not None or app.builder.format != "html":
        return
    source = _path_from_config(app, app.config.cpp_api_tagfile[0]).parent / "html"
    shutil.copytree(source, Path(app.outdir) / "cpp", dirs_exist_ok=True)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register Doxygen tag-file-backed cross-reference domains.

    Returns:
        Metadata declaring that the extension supports parallel builds.
    """
    app.add_config_value("cpp_api_tagfile", ("_build/doxygen/mqt-core.tag", "cpp/", "_build/doxygen/xml"), "env")
    app.add_config_value("qdmi_api_tagfile", ("_tagfiles/qdmi-1.3.2.tag", ""), "env")
    app.add_config_value(
        "qdmi_api_tagfile_url",
        "https://munich-quantum-software-stack.github.io/QDMI/v1.3.2/qdmi.tag",
        "env",
    )
    app.add_domain(CppApiDomain)
    app.add_domain(QdmiApiDomain)
    app.connect("builder-inited", build_doxygen)
    app.connect("build-finished", publish_doxygen_html)
    return {"parallel_read_safe": False, "parallel_write_safe": True}

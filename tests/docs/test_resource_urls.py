"""Documentation mirrors must update both hyperlinks and executable snippets."""

import runpy
from types import SimpleNamespace

import pytest

from tests.paths import REPO_ROOT

ConfigError = pytest.importorskip("sphinx.errors").ConfigError


@pytest.fixture
def config_module():
    return runpy.run_path(str(REPO_ROOT / "docs/source/conf.py"))


def test_default_resource_root(config_module):
    assert config_module["gsmap_resource_base_url"] == (
        "https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1"
    )


def test_override_normalizes_trailing_slash(config_module):
    app = SimpleNamespace(
        config=SimpleNamespace(gsmap_resource_base_url="https://mirror.example/gsMap/")
    )
    config_module["_configure_resource_url"](app, app.config)
    assert app.config.gsmap_resource_base_url == "https://mirror.example/gsMap"


@pytest.mark.parametrize(
    "page",
    [
        "downloads.md",
        "2d_tutorial/10x.md",
        "2d_tutorial/mouse_embryo.md",
        "3d_tutorial/human_gastrulation.md",
        "3d_tutorial/mouse_e11.md",
        "ldscore_weight_matrix.md",
    ],
)
def test_rebase_all_downloads_and_previews(config_module, page):
    original = (REPO_ROOT / "docs/source" / page).read_text()
    old = config_module["_source_resource_base_url"]
    new = "https://mirror.example/resources"
    app = SimpleNamespace(config=SimpleNamespace(gsmap_resource_base_url=new))
    source = [original]
    config_module["_replace_resource_urls"](app, page, source)
    assert f"{old}/" not in source[0]
    assert source[0].count(f"{new}/") == original.count(f"{old}/") > 0
    assert source[0].replace(f"{new}/", f"{old}/") == original


@pytest.mark.parametrize(
    "url",
    [
        "",
        "/relative",
        "ftp://mirror.example/data",
        "https://",
        "https://user:secret@mirror.example/data",
        "https://mirror.example/data?q=1",
        "https://mirror.example/data#fragment",
        "https://mirror.example/bad path",
        "https://mirror.example:bad/data",
        "https://mirror.example/$(bad command)",
        "https://mirror.example/`command`",
        'https://mirror.example/"bad"',
    ],
)
def test_invalid_resource_url_fails_early(config_module, url):
    config = SimpleNamespace(gsmap_resource_base_url=url)
    with pytest.raises(ConfigError, match="gsmap_resource_base_url"):
        config_module["_configure_resource_url"](None, config)

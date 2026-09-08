# Documentation tools

Run from the repository root after installing the documentation dependencies:

```bash
python -m docs.tools.generate_cli_reference
python -m docs.tools.generate_downloads --catalog docs/resource_catalog.json
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

These tools generate public documentation only. The resource catalogue contains
public download metadata, not local source mappings or server configuration.
Tutorial checks live in `tests/helpers/`; see `tests/README.md`.

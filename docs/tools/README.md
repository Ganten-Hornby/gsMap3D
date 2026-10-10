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

## Resource URL prefix

`gsmap_resource_base_url` in `docs/source/conf.py` controls download URLs,
tutorial `wget` commands and report-preview links in the built documentation.
Its default is
`https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1`.
To build for another server without editing the source pages:

```bash
sphinx-build -W --keep-going -b html \
    -D gsmap_resource_base_url=https://your-server.example/data/gsMap \
    docs/source docs/build/html
```

Alternatively, change the value directly in `conf.py`. Trailing slashes are
removed. Publication links and relative documentation links are unchanged.
The resource directory layout must be the same under the new prefix.
This configures documentation links only: it does not move files, update the
published catalogue or change the software's resource-download settings.

## Publishing to OBS

Future documentation and resource updates go directly to OBS, not the YangLab
Nginx filesystem. The local publish tree is
`/storage/yangjianLab/chenwenhao/resources/gsMap_public/publish/`; its root maps
to `s3://public/gsMap/gsMap3D-r1/`. Full setup, backup and validation instructions
are in `/storage/yangjianLab/chenwenhao/resources/gsMap_public/README.md`.

AWS CLI v2 is available at `/home/yangjianLab/chenwenhao/.local/bin/aws`,
independent of the gsMap Python environment. Use the `garage` profile with
region `garage`, path-style addressing and endpoint `http://10.0.3.164:3900`.
Credentials belong only in the user's private `~/.aws/credentials`, never in
this repository or public files. The private API endpoint is for uploads;
the HTTPS OBS prefix above is for readers.

Build into a fresh directory to avoid carrying obsolete pages into a release:

```bash
sphinx-build -E -a -W --keep-going -b html docs/source docs/build/obs-public-review
```

After validation, back up the previous local `publish/docs/r1/` and replace
that local directory with the public build outputs; do not copy `.doctrees/`,
`.buildinfo` or other internal build caches. Then inspect a documentation-only dry-run:

```bash
aws s3 sync \
    /storage/yangjianLab/chenwenhao/resources/gsMap_public/publish/docs/r1/ \
    s3://public/gsMap/gsMap3D-r1/docs/r1/ \
    --profile garage --endpoint-url http://10.0.3.164:3900 \
    --no-follow-symlinks --no-progress --dryrun
```

Remove `--dryrun` only for an intended release, after preserving any remote
contents that will be overwritten. Do not add `--delete` by default. Save the
upload log and exit status; check public page/asset URLs and file checksums.
Do not claim publication from a build or dry-run alone. Old YangLab receipts
are historical evidence, not proof of a new OBS publication.

## Performance figures

Regenerate the CPU and GPU scalability figures from the checked-in measurement
CSVs without running a new benchmark:

```bash
python -m docs.tools.plot_trait_scalability
python -m docs.tools.plot_scalability
```

`plot_trait_scalability` draws the IQ-only/four-trait CPU and GPU comparisons.
`plot_scalability` redraws the separate CPU batch-size sweep and the earlier
four-trait GPU sweep; the latter is retained as historical measurement data,
not mixed into the new curves. Both tools read local CSVs and need no GWAS files.

To import a new **complete, numerically validated** single-/four-trait sweep,
pass its results directory to `plot_trait_scalability --benchmark-root PATH`.
It requires all 72 measurements: CPU 1/2/4/8/16/32 cores (three repetitions),
and A100/V100/RTX 2080 Ti 1/2/4 GPUs (two repetitions), for both trait counts.
Both platforms must use the same 1,000,037 cells. CPU summaries are read from
`cpu/runs_million/`; GPU summaries from `gpu/scaling_results/`. Source and input
provenance must match across all summaries.
Incomplete or failed summaries cannot replace the public measurement CSV.

Measurement scope and provenance are recorded in
`docs/source/_static/benchmarks/README.md`. Do not mix different timing protocols
in a CPU-versus-GPU speedup comparison.

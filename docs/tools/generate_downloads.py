"""Render the public download page from validated artifact metadata."""

import argparse
import json
from pathlib import Path


def download_link(entry):
    url = entry["url"]
    return f"[Archive]({url}) · [SHA-256]({url}.sha256)"


def download_size(entry):
    size = entry["bytes"]
    return f"{size / (1024**3):.1f} GiB" if size >= 1024**3 else f"{size / (1024**2):.1f} MiB"


def render_catalog(catalog):
    titles = {
        "mouse-visium": "Mouse brain and kidney (Visium)",
        "mouse-e16-mosta": "Mouse E16.5 embryo (MOSTA)",
        "human-gastrulation-cs8": "Human CS8 gastrulation",
        "mouse-e11": "Mouse E11.5 embryo",
        "example-traits": "Six example GWAS",
        "homologs": "Mouse/macaque–human homologs",
        "additional-annotations": "Additional annotations",
        "ldscore-tutorial": "LD-matrix tutorial inputs",
        "ci-test-data": "Integration-test data",
    }
    sections = [
        ("2D example datasets", ("datasets/st/2d",)),
        ("3D example datasets", ("datasets/st/3d",)),
        ("Example GWAS", ("gwas",)),
        ("SNP × gene LD-weight matrices", ("reference/ld_weights",)),
        ("Homologs and additional annotations", ("reference/homologs", "reference/annotations")),
        ("Other resources", ("reference/ld_panels", "tutorials")),
    ]
    lines = []
    for title, categories in sections:
        entries = [entry for entry in catalog["resources"] if entry["category"] in categories]
        if not entries:
            continue
        lines.extend([f"### {title}", ""])
        if categories == ("reference/ld_weights",):
            lines.extend(
                [
                    "```{important}",
                    "Choose ancestry from the **GWAS population**, not the tissue species.",
                    "Each archive contains the SNP × gene matrix and its matching `w_ld/` files; use them together.",
                    "```",
                    "",
                ]
            )
            groups = {}
            for entry in entries:
                key = (entry["gencode"], entry["genome_build"])
                groups.setdefault(key, []).append(entry)
            order = {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}
            for (gencode, build), group in groups.items():
                lines.extend(
                    [
                        f"#### GENCODE {gencode} · {build}",
                        "",
                        "| Ancestry | Reference N | Size | Download |",
                        "|---|---:|---:|---|",
                    ]
                )
                for entry in sorted(group, key=lambda e: order.get(e["ancestry"], 99)):
                    lines.append(
                        f"| {entry['ancestry']} | {entry.get('reference_n', '—')} | "
                        f"{download_size(entry)} | {download_link(entry)} |"
                    )
                lines.append("")
            lines.extend(
                [
                    "V50 uses unrelated 1000G high-coverage panels, exact ±1 cM windows and MAC > 5.",
                    "Broad-reference SNPs retain MHC; regression SNPs and regression weights exclude it.",
                    "V46 uses the GRCh37 1000 Genomes Phase 3 low-coverage EUR reference panel, "
                    "paired with its original `weights_hm3_no_hla`. Do not mix the pairs.",
                    "[1000 Genomes reference releases](https://www.internationalgenome.org/announcements/3202-samples-at-high-coverage-from-NYGC/).",
                    "",
                ]
            )
            continue
        if categories in (("datasets/st/2d",), ("datasets/st/3d",)):
            lines.extend(
                [
                    "| Dataset | Spots / cells | Publication / source | Report preview | Download |",
                    "|---|---:|---|---|---|",
                ]
            )
            for entry in entries:
                # Display metadata can evolve without repacking immutable data archives.
                display = entry["display"]
                tutorial = entry["tutorial"].removesuffix(".html") + ".md"
                name = f"[{titles[entry['id']]}]({tutorial})<br>{display['description']}"
                sources = "<br>".join(
                    f"[{source['label']}]({source['url']})" for source in display["sources"]
                )
                previews = "<br>".join(
                    f"[{preview['label']}]({preview['url']})"
                    for preview in display["report_previews"]
                )
                count = f"{display['n_spots']:,}"
                if "spot_breakdown" in display:
                    count += f"<br>{display['spot_breakdown']}"
                lines.append(
                    f"| {name} | {count} | {sources} | {previews} | "
                    f"{download_link(entry)}<br>{download_size(entry)} |"
                )
            lines.extend(
                [
                    "",
                    "Counts are the spots or cells in the downloadable inputs, before gsMap QC.",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                "| Resource | Description | Size | Download |",
                "|---|---|---:|---|",
            ]
        )
        for entry in entries:
            description = entry["description"]
            if entry["id"] == "ldscore-tutorial":
                description = (
                    description.rstrip(".")
                    + "; tissue-specific ABC/Roadmap enhancer–gene BED files."
                )
            if "publication" in entry:
                description += f" [Publication]({entry['publication']})."
            if "tutorial" in entry:
                tutorial = entry["tutorial"].removesuffix(".html") + ".md"
                description += f" [Tutorial]({tutorial})."
            name = titles.get(entry["id"], entry["id"])
            lines.append(
                f"| {name} | {description} | {download_size(entry)} | {download_link(entry)} |"
            )
        lines.append("")
    lines.extend(
        [
            "Each bundle includes a README, metadata and `SHA256SUMS`. "
            "Tutorials include links to their report previews.",
            "See [Customization](advanced_usage.md) for using other paths or reference pairs.",
            "",
        ]
    )
    return "\n".join(lines)


def render(catalog):
    """One public Downloads page: authored setup plus generated artifact tables."""
    template = Path(__file__).with_name("templates") / "downloads.md"
    text = template.read_text()
    marker = "<!-- RESOURCE_CATALOG -->"
    if text.count(marker) != 1:
        raise ValueError("Downloads template needs exactly one resource catalog marker")
    return text.replace(marker, render_catalog(catalog)).rstrip() + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/source/downloads.md"))
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.catalog.read_text())))

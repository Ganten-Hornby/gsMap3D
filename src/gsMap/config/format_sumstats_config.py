"""
Configuration for formatting GWAS summary statistics.
"""

from dataclasses import dataclass
from typing import Optional, Annotated, Literal
import typer

@dataclass
class FormatSumstatsConfig:
    """Configuration for formatting GWAS summary statistics."""
    
    sumstats: Annotated[str, typer.Option(help="Path to gwas summary data")]
    out: Annotated[str, typer.Option(help="Path to save the formatted gwas data")]
    
    # Arguments for specify column name
    snp: Annotated[Optional[str], typer.Option(help="Name of snp column")] = None
    a1: Annotated[Optional[str], typer.Option(help="Name of effect allele column")] = None
    a2: Annotated[Optional[str], typer.Option(help="Name of none-effect allele column")] = None
    info: Annotated[Optional[str], typer.Option(help="Name of info column")] = None
    beta: Annotated[Optional[str], typer.Option(help="Name of gwas beta column.")] = None
    se: Annotated[Optional[str], typer.Option(help="Name of gwas standar error of beta column")] = None
    p: Annotated[Optional[str], typer.Option(help="Name of p-value column")] = None
    frq: Annotated[Optional[str], typer.Option(help="Name of A1 ferquency column")] = None
    n: Annotated[Optional[str], typer.Option(help="Name of sample size column")] = None
    z: Annotated[Optional[str], typer.Option(help="Name of gwas Z-statistics column")] = None
    OR: Annotated[Optional[str], typer.Option(help="Name of gwas OR column")] = None
    se_OR: Annotated[Optional[str], typer.Option(help="Name of standar error of OR column")] = None
    
    # Arguments for convert SNP (chr, pos) to rsid
    chr: Annotated[str, typer.Option(help="Name of SNP chromosome column")] = "Chr"
    pos: Annotated[str, typer.Option(help="Name of SNP positions column")] = "Pos"
    dbsnp: Annotated[Optional[str], typer.Option(help="Path to reference dnsnp file")] = None
    chunksize: Annotated[int, typer.Option(help="Chunk size for loading dbsnp file")] = 1000000
    
    # Arguments for output format and quality
    format: Annotated[Literal["gsMap", "COJO"], typer.Option(help="Format of output data", case_sensitive=False)] = "gsMap"
    info_min: Annotated[float, typer.Option(help="Minimum INFO score.")] = 0.9
    maf_min: Annotated[float, typer.Option(help="Minimum MAF.")] = 0.01
    keep_chr_pos: Annotated[bool, typer.Option(help="Keep SNP chromosome and position columns in the output data")] = False

    def __post_init__(self):
        # Handle n being potentially a number passed as a string from CLI
        if isinstance(self.n, str):
            try:
                if "." in self.n:
                    self.n = float(self.n)
                else:
                    self.n = int(self.n)
            except ValueError:
                # Leave as string if it's a column name
                pass

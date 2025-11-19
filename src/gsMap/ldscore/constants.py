"""
Constants and magic numbers used in LD score regression and spatial LDSC.

This module centralizes all numerical constants to improve code readability
and maintainability.
"""

# === LD Score Calculation Constants ===

# Degrees of freedom for unbiased L2 estimator bias correction
# The unbiased estimator is: r^2 - (1 - r^2) / (N - LDSC_BIAS_CORRECTION_DF)
LDSC_BIAS_CORRECTION_DF = 2.0

# Default LD window sizes
DEFAULT_LD_WINDOW_CM = 1.0  # centiMorgans
DEFAULT_LD_WINDOW_KB = 1000  # kilobases
DEFAULT_LD_WINDOW_MB = 1.0  # megabases (in base pairs)

# === Genomic Control Constants ===

# Median of chi-squared distribution with 1 degree of freedom
# Used for calculating genomic control lambda (λGC):
# λGC = median(χ²) / CHI_SQUARED_1DF_MEDIAN
# where χ² = Z^2 from GWAS summary statistics
CHI_SQUARED_1DF_MEDIAN = 0.4559364

# === Statistical Thresholds ===

# Standard genome-wide significance level (GWAS)
GWAS_SIGNIFICANCE_ALPHA = 5e-8

# Standard nominal significance level
NOMINAL_SIGNIFICANCE_ALPHA = 0.05

# FDR significance threshold (commonly used for spatial analysis)
FDR_SIGNIFICANCE_ALPHA = 0.001

# === Numerical Stability Constants ===

# Minimum p-value for log transformation (prevents log(0) = -inf)
# log10(1e-300) ≈ -300, which is reasonable for visualization
MIN_P_VALUE = 1e-300

# Maximum p-value (ceiling for numerical stability)
MAX_P_VALUE = 1.0

# Minimum MAF (minor allele frequency) for SNP filtering
DEFAULT_MAF_THRESHOLD = 0.05

# === Storage and Precision Constants ===

# Default dtype for LD score storage (saves ~75% disk space vs float32)
LDSCORE_STORAGE_DTYPE = "float16"

# Default dtype for computation
LDSCORE_COMPUTE_DTYPE = "float32"

# === Regression Constants ===

# Default number of jackknife blocks for standard error estimation
DEFAULT_N_BLOCKS = 200

# Minimum number of SNPs required for reliable LD score estimation
MIN_SNPS_FOR_LDSC = 200

# === Window Quantization Constants ===

# Default number of bins for dynamic programming quantization
# (balances JIT compilation overhead vs padding waste)
DEFAULT_QUANTIZATION_BINS = 20

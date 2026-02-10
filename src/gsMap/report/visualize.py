import gc
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import distinctipy
import matplotlib
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from rich import print
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial import KDTree
from tqdm import tqdm

warnings.filterwarnings("ignore")


def remove_outliers_MAD(data, threshold=3.5):
    """
    Remove outliers based on Median Absolute Deviation (MAD).
    """
    if isinstance(data, pd.Series | pd.DataFrame):
        data_values = data.values.flatten()
    else:
        data_values = np.asarray(data).flatten()

    if len(data_values) == 0:
        return data, np.ones(len(data), dtype=bool)

    median = np.nanmedian(data_values)
    mad = np.nanmedian(np.abs(data_values - median))
    if mad == 0:
        return data, np.ones(len(data), dtype=bool)

    modified_z_score = 0.6745 * (data_values - median) / mad
    mask = np.abs(modified_z_score) <= threshold

    if isinstance(data, pd.Series):
        return data[mask], mask
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            return data[mask], mask
        else:
            return data.flatten()[mask], mask
    return data[mask], mask


def load_ldsc(ldsc_input_file):
    ldsc = pd.read_csv(
        ldsc_input_file,
        compression="gzip",
        dtype={"spot": str, "p": float},
        index_col="spot",
        usecols=["spot", "p"],
    )
    ldsc["logp"] = -np.log10(ldsc.p)
    return ldsc


# %%
def load_st_coord(adata, feature_series: pd.Series, annotation):
    spot_name = adata.obs_names.to_list()
    assert "spatial" in adata.obsm.keys(), "spatial coordinates are not found in adata.obsm"

    # to DataFrame
    space_coord = adata.obsm["spatial"]
    if isinstance(space_coord, np.ndarray):
        space_coord = pd.DataFrame(space_coord, columns=["sx", "sy"], index=spot_name)
    else:
        space_coord = pd.DataFrame(space_coord.values, columns=["sx", "sy"], index=spot_name)

    space_coord = space_coord[space_coord.index.isin(feature_series.index)]
    space_coord_concat = pd.concat([space_coord.loc[feature_series.index], feature_series], axis=1)
    space_coord_concat.head()
    if annotation is not None:
        annotation = pd.Series(
            adata.obs[annotation].values, index=adata.obs_names, name="annotation"
        )
        space_coord_concat = pd.concat([space_coord_concat, annotation], axis=1)
    return space_coord_concat


def estimate_plotly_point_size(coordinates, DEFAULT_PIXEL_WIDTH=1000):
    # Convert to numpy array if it's a DataFrame or other array-like object
    if hasattr(coordinates, "values"):
        coordinates = coordinates.values
    coordinates = np.asarray(coordinates)

    tree = KDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    avg_min_distance = np.median(distances[:, 1])
    # get the width and height of the plot
    width = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    height = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])

    scale_factor = DEFAULT_PIXEL_WIDTH / max(width, height)
    pixel_width = width * scale_factor
    pixel_height = height * scale_factor

    point_size = avg_min_distance * scale_factor

    return (pixel_width, pixel_height), point_size


def estimate_matplotlib_scatter_marker_size(
    ax: matplotlib.axes.Axes,
    coordinates: np.ndarray,
    x_limits: tuple | None = None,
    y_limits: tuple | None = None,
) -> float:
    """
    Estimates the appropriate marker size to make adjacent markers touch.

    This function calculates the size 's' for a square marker (in points^2)
    such that its diameter in the plot corresponds to the average distance
    to the nearest neighbor for each point in the dataset. It accounts for
    the plot's aspect ratio and final rendered dimensions.

    Args:
        ax (matplotlib.axes.Axes): The subplot object. The function will
            temporarily set its limits and aspect ratio to ensure the
            transformation from data units to display units is accurate.
        coordinates (np.ndarray): A NumPy array of shape (n, 2)
            containing the (x, y) coordinates of the points.
        x_limits (Optional[tuple]): Optional (min, max) tuple to override automatic x-axis limits.
        y_limits (Optional[tuple]): Optional (min, max) tuple to override automatic y-axis limits.

    Returns:
        float: The estimated marker size 's' (in points^2) for use with
               ax.scatter().
    """
    # 1. Set up the axes' properties to ensure accurate transformations.
    # The aspect ratio and data limits must be set to correctly
    # calculate the relationship between data units and display units (inches/points).
    ax.set_aspect("equal")

    # Use provided limits if available, otherwise calculate from data
    if x_limits is not None:
        x_data_min, x_data_max = x_limits
    else:
        x_data_min, x_data_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])

    if y_limits is not None:
        y_data_min, y_data_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    else:
        y_data_min, y_data_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])

    ax.set_xlim(x_data_min, x_data_max)
    ax.set_ylim(y_data_min, y_data_max)

    # Force a draw of the canvas to finalize the transformations.
    ax.figure.canvas.draw()

    # 2. Calculate the required marker radius in data units.
    # We find the average distance to the nearest neighbor for all points.
    # The desired radius is half of this distance.
    tree = KDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    radius_data = np.mean(distances[:, 1]) / 2

    # 3. Convert the data radius to display units (points).
    # This requires transforming the axes' bounding box from data coordinates
    # to display coordinates (pixels), then to physical units (inches).

    # Get the bounding box in display (pixel) coordinates
    x_display_min, _ = ax.transData.transform((x_data_min, y_data_min))
    x_display_max, _ = ax.transData.transform((x_data_max, y_data_max))

    # Convert the display coordinates to inches
    x_inch_min, _ = ax.figure.dpi_scale_trans.inverted().transform((x_display_min, 0))
    x_inch_max, _ = ax.figure.dpi_scale_trans.inverted().transform((x_display_max, 0))

    width_inch = x_inch_max - x_inch_min
    width_data = x_data_max - x_data_min

    # Calculate the radius in inches. This scales the data radius by the
    # ratio of the plot's physical width to its data width.
    # This works because the aspect ratio is 'equal'.
    radius_inch = (radius_data / width_data) * width_inch

    # Convert inches to points (1 inch = 72 points).
    radius_points = radius_inch * 72

    # 4. Calculate the marker size 's'.
    # For ax.scatter, 's' is the marker area in points^2.
    # For a square marker, the area is (side)^2, where side = 2 * radius.
    square_marker_size = (2 * radius_points) ** 2

    return square_marker_size * 1.2


def draw_scatter(
    space_coord_concat,
    title=None,
    fig_style: Literal["dark", "light"] = "light",
    point_size: int = None,
    width=800,
    height=600,
    annotation=None,
    color_by="logp",
    color_continuous_scale=None,
    plot_origin="upper",
):
    # Set theme based on fig_style
    if fig_style == "dark":
        px.defaults.template = "plotly_dark"
    else:
        px.defaults.template = "plotly_white"

    if color_continuous_scale is None:
        custom_color_scale = [
            (1, "#d73027"),  # Red
            (7 / 8, "#f46d43"),  # Red-Orange
            (6 / 8, "#fdae61"),  # Orange
            (5 / 8, "#fee090"),  # Light Orange
            (4 / 8, "#e0f3f8"),  # Light Blue
            (3 / 8, "#abd9e9"),  # Sky Blue
            (2 / 8, "#74add1"),  # Medium Blue
            (1 / 8, "#4575b4"),  # Dark Blue
            (0, "#313695"),  # Deep Blue
        ]
        custom_color_scale.reverse()
        color_continuous_scale = custom_color_scale

    # Create the scatter plot
    fig = px.scatter(
        space_coord_concat,
        x="sx",
        y="sy",
        color=color_by,
        symbol="annotation" if annotation is not None else None,
        title=title,
        color_continuous_scale=color_continuous_scale,
        range_color=[0, max(space_coord_concat[color_by])],
    )

    # Update marker size if specified
    if point_size is not None:
        fig.update_traces(marker=dict(size=point_size, symbol="circle"))

    # Update layout for figure size
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )

    # Adjusting the legend
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.0,
            font=dict(
                size=10,
            ),
        )
    )

    # Update colorbar to be at the bottom and horizontal
    fig.update_layout(
        coloraxis_colorbar=dict(
            orientation="h",  # Make the colorbar horizontal
            x=0.5,  # Center the colorbar horizontally
            y=-0.0,  # Position below the plot
            xanchor="center",  # Anchor the colorbar at the center
            yanchor="top",  # Anchor the colorbar at the top to keep it just below the plot
            len=0.75,  # Length of the colorbar relative to the plot width
            title=dict(
                text="-log10(p)" if color_by == "logp" else color_by,  # Colorbar title
                side="top",  # Place the title at the top of the colorbar
            ),
        )
    )
    # Remove gridlines, axis labels, and ticks
    fig.update_xaxes(
        showgrid=False,  # Hide x-axis gridlines
        zeroline=False,  # Hide x-axis zero line
        showticklabels=False,  # Hide x-axis tick labels
        title=None,  # Remove x-axis title
        scaleanchor="y",  # Link the x-axis scale to the y-axis scale
    )

    fig.update_yaxes(
        showgrid=False,  # Hide y-axis gridlines
        zeroline=False,  # Hide y-axis zero line
        showticklabels=False,  # Hide y-axis tick labels
        title=None,  # Remove y-axis title
        autorange="reversed" if plot_origin == "upper" else True,
    )

    # Adjust margins to ensure no clipping and equal axis ratio
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=10),  # Adjust margins to prevent clipping
    )

    # Adjust the title location and font size
    fig.update_layout(
        title=dict(
            y=0.98,
            x=0.5,  # Center the title horizontally
            xanchor="center",  # Anchor the title at the center
            yanchor="top",  # Anchor the title at the top
            font=dict(
                size=20  # Increase the title font size
            ),
        )
    )

    return fig


def _create_color_map(category_list: list, hex=False, rng=42) -> dict[str, tuple]:
    unique_categories = sorted(set(category_list), key=str)

    # Check for 'NaN' or nan and handle separately
    nan_values = [v for v in unique_categories if str(v).lower() in ["nan", "none", "null"]]
    other_categories = [v for v in unique_categories if v not in nan_values]

    n_colors = len(other_categories)

    # Generate N visually distinct colors for non-NaN categories
    if n_colors > 0:
        colors = distinctipy.get_colors(n_colors, rng=rng)
        color_map = dict(zip(other_categories, colors, strict=False))
    else:
        color_map = {}

    # Assign grey color to NaN values
    grey_rgb = (0.827, 0.827, 0.827)  # lightgrey
    for v in nan_values:
        color_map[v] = grey_rgb

    if hex:
        # Convert RGB tuples to hex format
        color_map = {category: distinctipy.get_hex(color_map[category]) for category in color_map}
        print("Generated color map in hex format")
    return color_map


class VisualizeRunner:
    def __init__(self, config):
        self.config = config

    custom_colors_list = [
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695",
    ]

    def _generate_visualizations(self, obs_ldsc_merged: pd.DataFrame):
        """Generate all visualizations"""

        # Create visualization directories
        single_sample_folder = (
            self.config.visualization_result_dir / "single_sample_multi_trait_plot"
        )
        annotation_folder = self.config.visualization_result_dir / "annotation_distribution"
        annotation_folder.mkdir(exist_ok=True, parents=True)

        sample_names_list = sorted(obs_ldsc_merged["sample_name"].unique())

        for sample_name in tqdm(sample_names_list, desc="Generating visualizations"):
            # Multi-trait plot
            traits_png = (
                single_sample_folder / "static_png" / f"{sample_name}_gwas_traits_pvalues.jpg"
            )
            traits_pdf = (
                single_sample_folder / "static_pdf" / f"{sample_name}_gwas_traits_pvalues.pdf"
            )  # Added PDF output path

            # Create parent directories for the output files
            traits_png.parent.mkdir(exist_ok=True, parents=True)
            traits_pdf.parent.mkdir(exist_ok=True, parents=True)

            # Call the modified matplotlib-based plotting function.
            # This function saves files directly and does not return a figure object.
            self._create_single_sample_multi_trait_plots(
                obs_ldsc_merged=obs_ldsc_merged,
                trait_names=self.config.trait_name_list,
                sample_name=sample_name,
                output_png_path=traits_png,
                output_pdf_path=traits_pdf,
                max_cols=self.config.single_sample_multi_trait_max_cols,
                subsample_n_points=self.config.subsample_n_points,
                # Use new parameters from the updated VisualizationConfig
                subplot_width_inches=self.config.single_sample_multi_trait_subplot_width_inches,
                dpi=self.config.single_sample_multi_trait_dpi,
                enable_pdf_output=self.config.enable_pdf_output,
            )

            # Annotation distribution plots
            sample_data = obs_ldsc_merged.query(f'sample_name == "{sample_name}"')
            (pixel_width, pixel_height), point_size = estimate_plotly_point_size(
                sample_data[["sx", "sy"]].values
            )

            for annotation in self.config.cauchy_annotations:
                annotation_dir = annotation_folder / annotation
                annotation_dir.mkdir(exist_ok=True)

                annotation_color_map = _create_color_map(
                    obs_ldsc_merged[annotation].unique(), hex=True
                )
                fig = self._draw_scatter(
                    sample_data,
                    title=f"{annotation}_{sample_name}",
                    point_size=point_size,
                    width=pixel_width,
                    height=pixel_height,
                    hover_text_list=self.config.hover_text_list,
                    color_by=annotation,
                    color_map=annotation_color_map,
                )

                annotation_png = annotation_dir / f"{sample_name}_{annotation}.png"
                annotation_html = annotation_dir / f"{sample_name}_{annotation}.html"
                fig.write_image(annotation_png)
                fig.write_html(annotation_html)

        # Generate multi-sample annotation plots
        print("Generating multi-sample annotation plots...")
        sample_count = len(sample_names_list)
        n_rows, n_cols = self._calculate_optimal_grid_layout(
            item_count=sample_count, max_cols=self.config.single_sample_multi_trait_max_cols
        )

        for annotation in tqdm(
            self.config.cauchy_annotations, desc="Generating multi-sample annotation plots"
        ):
            annotation_dir = annotation_folder / annotation
            annotation_dir.mkdir(exist_ok=True)

            self._create_multi_sample_annotation_plot(
                obs_ldsc_merged=obs_ldsc_merged,
                annotation=annotation,
                sample_names_list=sample_names_list,
                output_dir=annotation_dir,
                n_rows=n_rows,
                n_cols=n_cols,
            )

    def _create_single_trait_multi_sample_plots(self, obs_ldsc_merged: pd.DataFrame):
        """Generate single trait multi-sample visualizations using matplotlib"""

        trait_names = self.config.trait_name_list

        # Create output directory
        single_trait_folder = (
            self.config.visualization_result_dir / "single_trait_multi_sample_plot"
        )
        single_trait_folder.mkdir(exist_ok=True, parents=True)

        # Prepare coordinate columns (assuming sx, sy are the spatial coordinates)
        obs_ldsc_merged = obs_ldsc_merged.copy()

        # Get sample count to determine grid dimensions
        sample_count = obs_ldsc_merged["sample_name"].nunique()
        n_rows, n_cols = self._calculate_optimal_grid_layout(
            item_count=sample_count, max_cols=self.config.single_trait_multi_sample_max_cols
        )

        print(
            f"Generating plots for {len(trait_names)} traits with {sample_count} samples in {n_rows}x{n_cols} grid"
        )

        # Generate visualization for each trait
        for trait in tqdm(trait_names, desc="Generating single trait multi-sample plots"):
            if trait not in obs_ldsc_merged.columns:
                print(f"Warning: Trait {trait} not found in data. Skipping.")
                continue

            self._create_single_trait_multi_sample_matplotlib_plot(
                obs_ldsc_merged=obs_ldsc_merged,
                trait_abbreviation=trait,
                output_png_path=single_trait_folder / f"{trait}_multi_sample_plot.jpg",
                output_pdf_path=single_trait_folder / f"{trait}_multi_sample_plot.pdf",
                n_rows=n_rows,
                n_cols=n_cols,
                subplot_width_inches=self.config.single_trait_multi_sample_subplot_width_inches,
                scaling_factor=self.config.single_trait_multi_sample_scaling_factor,
                dpi=self.config.single_trait_multi_sample_dpi,
                enable_pdf_output=self.config.enable_pdf_output,
                share_coords=self.config.share_coords,
            )

    def _calculate_optimal_grid_layout(
        self, item_count: int, max_cols: int = 8
    ) -> tuple[int, int]:
        """
        Calculate optimal grid dimensions (rows, cols) for displaying items in a grid.

        Args:
            item_count: Number of items to display
            max_cols: Maximum number of columns allowed

        Returns:
            tuple: (n_rows, n_cols) for optimal grid layout
        """
        import math

        if item_count <= 0:
            return 1, 1

        # For small counts, use simple layouts favoring horizontal arrangement
        if item_count <= 3:
            return 1, item_count
        elif item_count <= 6:
            return 2, math.ceil(item_count / 2)
        elif item_count <= 12:
            return 3, math.ceil(item_count / 3)
        else:
            # For larger counts, try to create a roughly square grid
            # but respect the max_cols constraint
            optimal_cols = min(math.ceil(math.sqrt(item_count)), max_cols)
            optimal_rows = math.ceil(item_count / optimal_cols)

            # If we hit the max_cols limit, recalculate rows
            if optimal_cols >= max_cols:
                n_cols = max_cols
                n_rows = math.ceil(item_count / max_cols)
            else:
                n_rows = optimal_rows
                n_cols = optimal_cols

        print(f"Calculated grid layout: {n_rows} rows × {n_cols} cols for {item_count} items")
        return n_rows, n_cols

    def _create_single_trait_multi_sample_matplotlib_plot(
        self,
        obs_ldsc_merged: pd.DataFrame,
        trait_abbreviation: str,
        sample_name_list: list[str] | None = None,
        output_png_path: Path | None = None,
        output_pdf_path: Path | None = None,
        n_rows: int = 6,
        n_cols: int = 8,
        subplot_width_inches: float = 4.0,
        scaling_factor: float = 1.0,
        dpi: int = 300,
        enable_pdf_output: bool = True,
        show=False,
        share_coords: bool = False,
    ):
        """
        Create and save a visualization for a specific trait showing all samples
        """

        matplotlib.rcParams["figure.dpi"] = dpi
        print(f"Creating visualization for {trait_abbreviation}")

        # Check if trait exists in the dataframe
        if trait_abbreviation not in obs_ldsc_merged.columns:
            print(f"Warning: Trait {trait_abbreviation} not found in the data. Skipping.")
            return

        # Set font to Arial with fallbacks to avoid warnings
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ]

        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", self.custom_colors_list
        )
        custom_cmap = custom_cmap.reversed()

        # Calculate figure size based on subplot dimensions
        fig_width = n_cols * subplot_width_inches
        fig_height = n_rows * subplot_width_inches

        # Create figure with title
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Add main title
        fig.suptitle(trait_abbreviation, fontsize=24, fontweight="bold", y=0.98)

        # Create grid of subplots
        grid_specs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, wspace=0.1, hspace=0.1)

        _, pass_filter_mask = remove_outliers_MAD(obs_ldsc_merged[trait_abbreviation])
        obs_ldsc_merged_filtered = obs_ldsc_merged[pass_filter_mask]

        pd_min = 0
        pd_max = obs_ldsc_merged_filtered[trait_abbreviation].quantile(0.999)

        print(f"Color scale min: {pd_min}, max: {pd_max}")
        # Get list of sample names - use provided list or fallback to sorted unique
        if sample_name_list is None:
            sample_name_list = sorted(obs_ldsc_merged_filtered["sample_name"].unique())

        # get the x and y limit if share coordinates
        if share_coords:
            x_limits = (obs_ldsc_merged_filtered["sx"].min(), obs_ldsc_merged_filtered["sx"].max())
            y_limits = (obs_ldsc_merged_filtered["sy"].min(), obs_ldsc_merged_filtered["sy"].max())
        else:
            x_limits = None
            y_limits = None

        # Create a scatter plot for each sample
        for position_num, select_sample_name in enumerate(sample_name_list[: n_rows * n_cols], 1):
            # Calculate row and column in the grid
            row = (position_num - 1) // n_cols
            col = (position_num - 1) % n_cols

            # Create subplot
            ax = fig.add_subplot(grid_specs[row, col])

            # Get data for this sample
            sample_data = obs_ldsc_merged_filtered[
                obs_ldsc_merged_filtered["sample_name"] == select_sample_name
            ]

            point_size = self.estimate_matplitlib_scatter_marker_size(
                ax, sample_data[["sx", "sy"]].values, x_limits=x_limits, y_limits=y_limits
            )
            point_size *= scaling_factor  # Apply scaling factor
            # Create scatter plot
            scatter = ax.scatter(
                sample_data["sx"],
                sample_data["sy"],
                c=sample_data[trait_abbreviation],
                cmap=custom_cmap,
                s=point_size,
                vmin=pd_min,
                vmax=pd_max,
                marker="o",
                edgecolors="none",
                rasterized=True if output_pdf_path is not None and enable_pdf_output else False,
            )

            if self.config.plot_origin == "upper":
                ax.invert_yaxis()

            ax.axis("off")
            # Add sample label as title
            ax.set_title(select_sample_name, fontsize=12, pad=None)

        # Add colorbar to the right side
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("$-\\log_{10}p$", fontsize=12, fontweight="bold")

        if output_png_path is not None:
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_png_path,
                dpi=dpi,
                bbox_inches="tight",
            )

        if output_pdf_path is not None and enable_pdf_output:
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_pdf_path,
                bbox_inches="tight",
            )

        # Close the figure to free memory only if not returning it
        if show:
            plt.show()

        gc.collect()
        return fig

    def _create_single_sample_multi_trait_plots(
        self,
        obs_ldsc_merged: pd.DataFrame,
        trait_names: list[str],
        sample_name: str,
        # New arguments for output paths, as matplotlib saves directly
        output_png_path: Path | None,
        output_pdf_path: Path | None,
        # Arguments from original function signature, adapted for matplotlib
        max_cols: int = 5,
        subsample_n_points: int | None = None,
        # subplot_width is now interpreted as inches for figsize
        subplot_width_inches: float = 4.0,
        dpi: int = 300,
        enable_pdf_output: bool = True,
    ):
        print(f"Creating Matplotlib-based multi-trait visualization for sample: {sample_name}")

        # 1. Filter data for the specific sample and subsample if requested
        sample_plot_data = obs_ldsc_merged[obs_ldsc_merged["sample_name"] == sample_name].copy()
        if subsample_n_points and len(sample_plot_data) > subsample_n_points:
            print(f"Subsampling to {subsample_n_points} points for plotting.")
            sample_plot_data = sample_plot_data.sample(n=subsample_n_points, random_state=42)

        if sample_plot_data.empty:
            print(f"Warning: No data found for sample '{sample_name}'. Skipping plot generation.")
            return

        # 2. Calculate optimal grid layout for subplots
        n_traits = len(trait_names)
        n_rows, n_cols = self._calculate_optimal_grid_layout(
            item_count=n_traits, max_cols=max_cols
        )
        print(f"Plotting {n_traits} traits in a {n_rows}x{n_cols} grid.")

        # 3. Determine figure size and create figure and axes
        # Estimate subplot height based on data's aspect ratio to avoid distortion
        x_range = sample_plot_data["sx"].max() - sample_plot_data["sx"].min()
        y_range = sample_plot_data["sy"].max() - sample_plot_data["sy"].min()
        aspect_ratio = y_range / x_range if x_range > 0 else 1.0
        subplot_height_inches = subplot_width_inches * aspect_ratio

        # Calculate total figure size, adding padding for titles and colorbars
        fig_width = subplot_width_inches * n_cols
        fig_height = (
            subplot_height_inches * n_rows
        ) * 1.2  # Add 20% vertical space for titles/colorbars

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        fig.suptitle(f"Sample: {sample_name}", fontsize=16, fontweight="bold")

        # 4. Define custom colormap and font
        # Set font to Arial with fallbacks to avoid warnings
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ]

        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom_cmap", self.custom_colors_list
        ).reversed()

        # 5. Iterate through traits and create each subplot
        axes_flat = axes.flatten()
        for i, trait in enumerate(trait_names):
            if i >= len(axes_flat):
                break  # Should not happen with correct grid calculation, but is a safe guard

            ax = axes_flat[i]

            # Estimate marker size to fill space without overlap
            point_size = estimate_matplotlib_scatter_marker_size(
                ax, sample_plot_data[["sx", "sy"]].values
            )

            # Determine color scale, capping at the 99.9th percentile to handle outliers
            sample_trait_data = sample_plot_data[["sx", "sy", trait]].dropna()
            trait_values, mask = remove_outliers_MAD(sample_trait_data[trait])
            sample_trait_data = sample_trait_data[mask]  # filter out outliers

            vmin = 0
            vmax = trait_values.quantile(0.999)
            if pd.isna(vmax) or vmax == 0:
                vmax = trait_values.max() if trait_values.max() > 0 else 1.0

            # Create the scatter plot
            scatter = ax.scatter(
                sample_trait_data["sx"],
                sample_trait_data["sy"],
                c=trait_values,
                cmap=custom_cmap,
                s=point_size,
                vmin=vmin,
                vmax=vmax,
                marker="o",
                edgecolors="none",
                rasterized=True if output_pdf_path is not None and enable_pdf_output else False,
            )

            ax.set_title(trait, fontsize=16, pad=10, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")

            if self.config.plot_origin == "upper":
                ax.invert_yaxis()

            ax.axis("off")

            # Add a colorbar to each subplot
            cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.1, fraction=0.05)
            cbar.set_label("$-\\log_{10}p$", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        # Hide any unused axes in the grid
        for j in range(len(trait_names), len(axes_flat)):
            axes_flat[j].axis("off")
        #
        # # 6. Adjust layout and save the figure
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle and bottom elements

        # Only proceed with saving if paths are provided
        if output_png_path is not None:
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved multi-trait plot for '{sample_name}' to:\n  - {output_png_path}")

        if output_pdf_path is not None and enable_pdf_output:
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_pdf_path, bbox_inches="tight", facecolor="white")
            print(f"Saved multi-trait plot for '{sample_name}' to:\n  - {output_pdf_path}")

        # Clean up to free memory
        plt.close(fig)

    def _draw_scatter(
        self,
        space_coord_concat: pd.DataFrame,
        title: str | None = None,
        fig_style: str = "light",
        point_size: int | None = None,
        hover_text_list: list[str] | None = None,
        width: int = 800,
        height: int = 600,
        annotation: str | None = None,
        color_by: str = "logp",
        color_map: dict | None = None,
    ):
        """Create scatter plot (adapted from original draw_scatter function)"""
        # Set theme based on fig_style
        if fig_style == "dark":
            px.defaults.template = "plotly_dark"
        else:
            px.defaults.template = "plotly_white"

        custom_color_scale = [
            (1, "#d73027"),  # Red
            (7 / 8, "#f46d43"),  # Red-Orange
            (6 / 8, "#fdae61"),  # Orange
            (5 / 8, "#fee090"),  # Light Orange
            (4 / 8, "#e0f3f8"),  # Light Blue
            (3 / 8, "#abd9e9"),  # Sky Blue
            (2 / 8, "#74add1"),  # Medium Blue
            (1 / 8, "#4575b4"),  # Dark Blue
            (0, "#313695"),  # Deep Blue
        ]
        custom_color_scale.reverse()

        # if category data
        if not pd.api.types.is_numeric_dtype(space_coord_concat[color_by]):
            # Create the scatter plot
            fig = px.scatter(
                space_coord_concat,
                x="sx",
                y="sy",
                color=color_by,
                # symbol=annotation,
                title=title,
                color_discrete_map=color_map,
                hover_name=color_by,
                hover_data=hover_text_list,
                # color_continuous_scale=custom_color_scale,
                # range_color=[0, max(space_coord_concat[color_by])],
            )
        else:
            fig = px.scatter(
                space_coord_concat,
                x="sx",
                y="sy",
                color=color_by,
                symbol=annotation,
                title=title,
                hover_name=color_by,
                hover_data=hover_text_list,
                color_continuous_scale=custom_color_scale,
                range_color=[0, space_coord_concat[color_by].max()],
            )

        # Update marker size if specified
        if point_size is not None:
            fig.update_traces(marker=dict(size=point_size, symbol="circle"))

        # Update layout for figure size
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
        )

        # Adjusting the legend - Updated position and marker size
        fig.update_layout(
            legend=dict(
                yanchor="middle",  # Anchor point for y
                y=0.5,  # Center vertically
                xanchor="left",  # Anchor point for x
                x=1.02,  # Position just outside the plot
                font=dict(
                    size=10,
                ),
                itemsizing="constant",  # Makes legend markers a constant size
                itemwidth=30,  # Adjust width of legend items
            )
        )

        # Update colorbar to be at the bottom and horizontal
        fig.update_layout(
            coloraxis_colorbar=dict(
                orientation="h",
                x=0.5,
                y=-0.0,
                xanchor="center",
                yanchor="top",
                len=0.75,
                title=dict(text="-log10(p)" if color_by == "logp" else color_by, side="top"),
            )
        )

        # Remove gridlines, axis labels, and ticks
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None,
            scaleanchor="y",
        )

        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None,
            autorange="reversed" if self.config.plot_origin == "upper" else True,
        )

        # Adjust margins to ensure no clipping and equal axis ratio
        fig.update_layout(
            margin=dict(l=0, r=100, t=20, b=10),  # Increased right margin to accommodate legend
            height=width,
        )

        # Adjust the title location and font size
        fig.update_layout(
            title=dict(y=0.98, x=0.5, xanchor="center", yanchor="top", font=dict(size=20))
        )

        return fig

    @classmethod
    def estimate_matplitlib_scatter_marker_size(
        cls,
        ax: matplotlib.axes.Axes,
        coordinates: np.ndarray,
        x_limits: tuple | None = None,
        y_limits: tuple | None = None,
    ) -> float:
        """Alias for estimate_matplotlib_scatter_marker_size (with typo) for backward compatibility."""
        return estimate_matplotlib_scatter_marker_size(ax, coordinates, x_limits, y_limits)

    @classmethod
    def estimate_matplotlib_scatter_marker_size(
        cls,
        ax: matplotlib.axes.Axes,
        coordinates: np.ndarray,
        x_limits: tuple | None = None,
        y_limits: tuple | None = None,
    ) -> float:
        """Alias for estimate_matplotlib_scatter_marker_size for backward compatibility."""
        return estimate_matplotlib_scatter_marker_size(ax, coordinates, x_limits, y_limits)

    def _create_multi_sample_annotation_plot(
        self,
        obs_ldsc_merged: pd.DataFrame,
        annotation: str,
        sample_names_list: list,
        output_dir: Path,
        n_rows: int,
        n_cols: int,
        fig_width: float = 20,
        fig_height: float = 15,
        scaling_factor: float = 1.0,
        dpi: int = 300,
    ):
        """Create multi-sample annotation plot using matplotlib with subplots for each sample"""

        print(f"Creating multi-sample plot for annotation: {annotation}")

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(f"{annotation} - All Samples", fontsize=24, fontweight="bold", y=0.98)

        # Create grid of subplots
        grid_specs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, wspace=0.1, hspace=0.1)

        # Get unique annotation values and create color map
        unique_annotations = obs_ldsc_merged[annotation].unique()
        if pd.api.types.is_numeric_dtype(obs_ldsc_merged[annotation]):
            custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "custom_cmap", self.custom_colors_list
            )
            cmap = custom_cmap.reversed()
            norm = plt.Normalize(
                vmin=obs_ldsc_merged[annotation].min(), vmax=obs_ldsc_merged[annotation].max()
            )
        else:
            # For categorical annotations, use discrete colors
            color_map = _create_color_map(unique_annotations, hex=False)

        # Create scatter plot for each sample
        for position_num, sample_name in enumerate(sample_names_list[: n_rows * n_cols], 1):
            # Calculate row and column in the grid
            row = (position_num - 1) // n_cols
            col = (position_num - 1) % n_cols

            # Create subplot
            ax = fig.add_subplot(grid_specs[row, col])

            # Get data for this sample
            sample_data = obs_ldsc_merged[obs_ldsc_merged["sample_name"] == sample_name]

            # Estimate point size based on data density
            point_size = estimate_matplotlib_scatter_marker_size(
                ax, sample_data[["sx", "sy"]].values
            )
            point_size *= scaling_factor  # Apply scaling factor

            # Create scatter plot
            if pd.api.types.is_numeric_dtype(obs_ldsc_merged[annotation]):
                ax.scatter(
                    sample_data["sx"],
                    sample_data["sy"],
                    c=sample_data[annotation],
                    cmap=cmap,
                    norm=norm,
                    s=point_size,
                    alpha=1.0,
                    edgecolors="none",
                )
            else:
                # For categorical data, plot each category separately
                for cat in unique_annotations:
                    cat_data = sample_data[sample_data[annotation] == cat]
                    if len(cat_data) > 0:
                        ax.scatter(
                            cat_data["sx"],
                            cat_data["sy"],
                            c=[color_map[cat]],
                            s=point_size,
                            alpha=1.0,
                            edgecolors="none",
                            label=cat,
                        )

            # Set subplot title
            ax.set_title(sample_name, fontsize=10)
            ax.set_aspect("equal")
            if self.config.plot_origin == "upper":
                ax.invert_yaxis()
            ax.axis("off")

        # Add colorbar for numeric annotations or legend for categorical
        if pd.api.types.is_numeric_dtype(obs_ldsc_merged[annotation]):
            # Create a colorbar on the right side of the figure
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
            cbar.set_label(annotation, fontsize=14)
        else:
            # Create a legend on the right side of the figure
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=color,
                    markersize=getattr(self.config, "legend_marker_size", 10),
                )
                for label, color in color_map.items()
            ]
            fig.subplots_adjust(right=0.8)
            fig.legend(
                handles=handles, title=annotation, loc="center left", bbox_to_anchor=(0.85, 0.5)
            )

        # Save the plot if output_dir is provided
        if output_dir:
            output_path = output_dir / f"multi_sample_{annotation}.png"
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved multi-sample annotation plot: {output_path}")

        return fig

    def _run_cauchy_analysis(self, obs_ldsc_merged: pd.DataFrame):
        """Run Cauchy combination analysis"""

        trait_names = self.config.trait_name_list

        for annotation_col in self.config.cauchy_annotations:
            print(f"Running Cauchy analysis for {annotation_col}...")

            cauchy_results = self._run_cauchy_combination_per_annotation(
                obs_ldsc_merged, annotation_col=annotation_col, trait_cols=trait_names
            )

            # Save results
            output_file = (
                self.config.visualization_result_dir / f"cauchy_results_{annotation_col}.csv"
            )
            self._save_cauchy_results_to_csv(cauchy_results, output_file)

            # Generate heatmaps
            self._generate_cauchy_heatmaps(cauchy_results, annotation_col)

    def _run_cauchy_combination_per_annotation(
        self, df: pd.DataFrame, annotation_col: str, trait_cols: list[str], max_workers=None
    ):
        """
        Runs the Cauchy combination on each annotation category for each given trait in parallel.
        Also calculates odds ratios with confidence intervals for significant spots in each annotation.
        """
        from functools import partial

        import statsmodels.api as sm
        from scipy.stats import fisher_exact

        results_dict = {}
        annotations = df[annotation_col].unique()

        # Helper function to process a single trait for a given annotation
        def process_trait(trait, anno_data, all_data, annotation):
            # Calculate significance threshold (Bonferroni correction)
            sig_threshold = 0.05 / len(all_data)

            # Get p-values for this annotation and trait
            log10p = anno_data[trait].values
            log10p, mask = remove_outliers_MAD(
                log10p,
            )
            p_values = 10 ** (-log10p)  # convert from log10(p) to p

            # Calculate Cauchy combination and median
            p_cauchy_val = self._acat_test(p_values)
            p_median_val = np.median(p_values)

            # Calculate significance statistics
            sig_spots_in_anno = np.sum(p_values < sig_threshold)
            total_spots_in_anno = len(p_values)

            # Get p-values for other annotations
            other_annotations_mask = all_data[annotation_col] != annotation
            other_p_values = 10 ** (-all_data.loc[other_annotations_mask, trait].values)
            sig_spots_elsewhere = np.sum(other_p_values < sig_threshold)
            total_spots_elsewhere = len(other_p_values)

            # Odds ratio calculation using Fisher's exact test
            try:
                # Create contingency table
                contingency_table = np.array(
                    [
                        [sig_spots_in_anno, total_spots_in_anno - sig_spots_in_anno],
                        [sig_spots_elsewhere, total_spots_elsewhere - sig_spots_elsewhere],
                    ]
                )

                # Calculate odds ratio and p-value using Fisher's exact test
                odds_ratio, p_value = fisher_exact(contingency_table)

                # if odds_ratio is infinite, set it to a large number
                if odds_ratio == np.inf:
                    odds_ratio = 1e4  # Set to a large number to avoid overflow

                # Calculate confidence intervals
                table = sm.stats.Table2x2(contingency_table)
                conf_int = table.oddsratio_confint()
                ci_low, ci_high = conf_int
            except Exception as e:  # noqa: BLE001
                # Handle calculation errors
                odds_ratio = 0
                p_value = 1
                ci_low, ci_high = 0, 0
                print(f"Fisher's exact test failed for {trait} in {annotation}: {e}")

            return {
                "trait": trait,
                "p_cauchy": p_cauchy_val,
                "p_median": p_median_val,
                "odds_ratio": odds_ratio,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_odds_ratio": p_value,
                "sig_spots": sig_spots_in_anno,
                "total_spots": total_spots_in_anno,
                "sig_ratio": sig_spots_in_anno / total_spots_in_anno
                if total_spots_in_anno > 0
                else 0,
                "overall_sig_spots": sig_spots_in_anno + sig_spots_elsewhere,
                "overall_spots": total_spots_in_anno + total_spots_elsewhere,
            }

        # Process each annotation (sequential)
        for anno in tqdm(annotations, desc="Processing annotations"):
            df_anno = df[df[annotation_col] == anno]

            # Create a partial function with fixed parameters
            process_trait_for_anno = partial(
                process_trait, anno_data=df_anno, all_data=df, annotation=anno
            )

            # Process traits in parallel with progress bar
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create list for results and submit all tasks
                futures = list(
                    tqdm(
                        executor.map(process_trait_for_anno, trait_cols),
                        total=len(trait_cols),
                        desc=f"Processing traits for {anno}",
                        leave=False,
                    )
                )
                trait_results = list(futures)

            # Create a DataFrame for this annotation
            anno_results_df = pd.DataFrame(trait_results).sort_values(by="p_cauchy")
            results_dict[anno] = anno_results_df

        return results_dict

    def _acat_test(self, pvalues: np.ndarray, weights=None):
        logger = logging.getLogger("gsMap.post_analysis.cauchy")
        if np.any(np.isnan(pvalues)):
            raise ValueError("Cannot have NAs in the p-values.")
        if np.any((pvalues > 1) | (pvalues < 0)):
            raise ValueError("P-values must be between 0 and 1.")
        if np.any(pvalues == 0) and np.any(pvalues == 1):
            raise ValueError("Cannot have both 0 and 1 p-values.")
        if np.any(pvalues == 0):
            logger.info("Warn: p-values are exactly 0.")
            return 0
        if np.any(pvalues == 1):
            logger.info("Warn: p-values are exactly 1.")
            return 1

        if weights is None:
            weights = np.full(len(pvalues), 1 / len(pvalues))
        else:
            if len(weights) != len(pvalues):
                raise Exception("Length of weights and p-values differs.")
            if any(weights < 0):
                raise Exception("All weights must be positive.")
            weights = np.array(weights) / np.sum(weights)

        is_small = pvalues < 1e-16
        is_large = ~is_small

        if not np.any(is_small):
            cct_stat = np.sum(weights * np.tan((0.5 - pvalues) * np.pi))
        else:
            cct_stat = np.sum((weights[is_small] / pvalues[is_small]) / np.pi) + np.sum(
                weights[is_large] * np.tan((0.5 - pvalues[is_large]) * np.pi)
            )

        if cct_stat > 1e15:
            pval = (1 / cct_stat) / np.pi
        else:
            pval = 1 - stats.cauchy.cdf(cct_stat)

        return pval

    def _save_cauchy_results_to_csv(self, cauchy_results: dict, output_path: Path):
        """Save Cauchy results to CSV"""
        all_results = []
        for annotation, df in cauchy_results.items():
            df_copy = df.copy()
            df_copy["annotation"] = annotation
            all_results.append(df_copy)

        combined_results = pd.concat(all_results, ignore_index=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_results.to_csv(output_path, index=False)

        return combined_results

    def _generate_cauchy_heatmaps(self, cauchy_results: dict, annotation_col: str):
        """Generate multiple types of Cauchy combination heatmaps"""
        # Convert results to pivot table format for different metrics
        table_cauchy = self._results_dict_to_log10_table(
            cauchy_results, value_col="p_cauchy", log10_transform=True
        )
        table_median = self._results_dict_to_log10_table(
            cauchy_results, value_col="p_median", log10_transform=True
        )
        table_odds_ratio = self._results_dict_to_log10_table(
            cauchy_results, value_col="odds_ratio", log10_transform=False
        )

        # Create heatmap directories
        cauchy_heatmap_base = self.config.visualization_result_dir / "cauchy_heatmap"
        static_folder = cauchy_heatmap_base / "static_png"
        interactive_folder = cauchy_heatmap_base / "interactive_html"

        for folder in [static_folder, interactive_folder]:
            folder.mkdir(exist_ok=True, parents=True)

        # Calculate dimensions
        num_annotations, num_traits = table_cauchy.shape
        width = 50 * num_traits
        height = 50 * num_annotations

        # 1. Cauchy combination heatmap (non-normalized)
        fig = self._plot_p_cauchy_heatmap(
            df=table_cauchy,
            title=f"Cauchy Combination Heatmap -- By {annotation_col}",
            cluster_rows=True,
            cluster_cols=True,
            width=width,
            height=height,
            text_format=".2f",
            font_size=10,
            margin_pad=150,
        )
        fig.write_image(static_folder / f"cauchy_combination_by_{annotation_col}.png", scale=2)
        fig.write_html(interactive_folder / f"cauchy_combination_by_{annotation_col}.html")

        # 2. Cauchy combination heatmap (normalized)
        fig = self._plot_p_cauchy_heatmap(
            df=table_cauchy,
            title=f"Cauchy Combination Heatmap -- By {annotation_col}",
            normalize_axis="column",
            cluster_rows=True,
            cluster_cols=True,
            width=width,
            height=height,
            text_format=".2f",
            font_size=10,
            margin_pad=150,
        )
        fig.write_image(
            static_folder / f"cauchy_combination_by_{annotation_col}_normalized.png", scale=2
        )
        fig.write_html(
            interactive_folder / f"cauchy_combination_by_{annotation_col}_normalized.html"
        )

        # 3. Median p-value heatmap (non-normalized)
        fig = self._plot_p_cauchy_heatmap(
            df=table_median,
            title=f"Median log 10 pvalue Heatmap -- By {annotation_col}",
            cluster_rows=True,
            cluster_cols=True,
            width=width,
            height=height,
            text_format=".2f",
            font_size=10,
            margin_pad=150,
        )
        fig.write_image(static_folder / f"median_pvalue_{annotation_col}.png", scale=2)
        fig.write_html(interactive_folder / f"median_pvalue_{annotation_col}.html")

        # 4. Median p-value heatmap (normalized)
        fig = self._plot_p_cauchy_heatmap(
            df=table_median,
            title=f"Median log 10 pvalue Heatmap -- By {annotation_col}",
            normalize_axis="column",
            cluster_rows=True,
            cluster_cols=True,
            width=width,
            height=height,
            text_format=".2f",
            font_size=10,
            margin_pad=150,
        )
        fig.write_image(static_folder / f"median_pvalue_{annotation_col}_normalized.png", scale=2)
        fig.write_html(interactive_folder / f"median_pvalue_{annotation_col}_normalized.html")

        # 5. Odds ratio heatmap
        fig = self._plot_p_cauchy_heatmap(
            df=table_odds_ratio,
            title=f"Odds Ratio Heatmap -- By {annotation_col}",
            cluster_rows=True,
            cluster_cols=True,
            width=width,
            height=height,
            text_format=".2f",
            font_size=10,
            margin_pad=150,
        )
        fig.write_image(static_folder / f"odds_ratio_{annotation_col}.png", scale=2)
        fig.write_html(interactive_folder / f"odds_ratio_{annotation_col}.html")

    def _results_dict_to_log10_table(
        self,
        results_dict: dict,
        value_col: str = "p_cauchy",
        log10_transform: bool = True,
        epsilon: float = 1e-300,
    ) -> pd.DataFrame:
        """Convert results dict to pivot table"""
        all_data = []
        for anno, df in results_dict.items():
            temp = df.copy()
            temp["annotation"] = anno
            all_data.append(temp)

        combined_df = pd.concat(all_data, ignore_index=True)

        if log10_transform:
            combined_df.loc[combined_df[value_col] == 0, value_col] = epsilon
            combined_df["transformed"] = -np.log10(combined_df[value_col])
        else:
            combined_df["transformed"] = combined_df[value_col]

        pivot_df = combined_df.pivot(index="annotation", columns="trait", values="transformed")
        return pivot_df

    def _plot_p_cauchy_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Cauchy Combination Heatmap",
        normalize_axis: Literal["row", "column"] | None = None,
        cluster_rows: bool = False,
        cluster_cols: bool = False,
        color_continuous_scale: str | list = "RdBu_r",
        width: int | None = None,
        height: int | None = None,
        text_format: str = ".2f",
        show_text: bool = True,
        font_size: int = 10,
        margin_pad: int = 150,
    ) -> go.Figure:
        """
        Create an enhanced heatmap visualization for trait-annotation relationships.
        """
        data = df.copy()

        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if not np.issubdtype(data.values.dtype, np.number):
            raise ValueError("DataFrame must contain numeric values")

        n_rows, n_cols = data.shape
        # Set dynamic width/height if not provided to ensure good aspect ratio
        # Previously we used 50 and 30, which led to vertical stretching.
        # Let's use more balanced units.
        if width is None:
            width = max(600, n_cols * 150 + margin_pad * 2)
        if height is None:
            height = max(500, n_rows * 60 + margin_pad * 2)

        # Normalization with error handling
        if normalize_axis in ["row", "column"]:
            axis = 1 if normalize_axis == "row" else 0
            try:
                # Store original data for text annotations
                original_data = data.copy()

                # Calculate min and max along specified axis
                min_vals = data.min(axis=axis)
                max_vals = data.max(axis=axis)
                range_vals = max_vals - min_vals

                # Replace zero range with 1 to avoid division by zero
                range_vals = range_vals.replace(0, 1)

                # Normalize using broadcasting
                if normalize_axis == "row":
                    data = data.sub(min_vals, axis=0).div(range_vals, axis=0)
                else:  # column
                    data = data.sub(min_vals, axis=1).div(range_vals, axis=1)

                data = data.fillna(0)
            except Exception as e:
                raise ValueError(f"Normalization failed: {str(e)}") from e
        else:
            # No normalization, use original data for both color and text
            original_data = data

        # Clustering with error handling
        try:
            if cluster_rows:
                row_linkage = linkage(data.fillna(0).values, method="average", metric="euclidean")
                row_order = leaves_list(row_linkage)
                data = data.iloc[row_order, :]
                original_data = original_data.iloc[
                    row_order, :
                ]  # Apply the same order to original data

            if cluster_cols:
                col_linkage = linkage(
                    data.fillna(0).values.T, method="average", metric="euclidean"
                )
                col_order = leaves_list(col_linkage)
                data = data.iloc[:, col_order]
                original_data = original_data.iloc[
                    :, col_order
                ]  # Apply the same order to original data
        except Exception as e:
            raise ValueError(f"Clustering failed: {str(e)}") from e

        # Create heatmap with enhanced formatting
        if normalize_axis is None:
            # Use original settings for speed when no normalization is applied
            fig = px.imshow(
                data,
                color_continuous_scale=color_continuous_scale,
                aspect="auto",
                width=width,
                height=height,
                text_auto=text_format if show_text else False,  # Automatic text generation
            )
        else:
            # Use custom logic for normalization (manual text annotations)
            fig = px.imshow(
                data,
                color_continuous_scale=color_continuous_scale,
                aspect="auto",
                width=width,
                height=height,
                text_auto=False,  # Disable automatic text generation
            )

            # Add manual text annotations using original data
            if show_text:
                for i, row in enumerate(original_data.values):
                    for j, value in enumerate(row):
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=f"{value:{text_format}}",
                            showarrow=False,
                            font=dict(size=font_size, color="black"),
                        )

        # Enhanced layout configuration
        fig.update_layout(
            title={
                "text": title,
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "bottom",
                "font": {"size": font_size + 4},
            },
            xaxis={
                "title": "Trait",
                "tickangle": 45,
                "side": "bottom",
                "tickfont": {"size": font_size},
                "title_font": {"size": font_size + 2},
            },
            yaxis={
                "title": "Annotation",
                "tickfont": {"size": font_size},
                "title_font": {"size": font_size + 2},
            },
            width=width,
            height=height,
            template="plotly_white",
            margin=dict(l=margin_pad, r=margin_pad, t=margin_pad, b=margin_pad),
            coloraxis_colorbar={
                "title": "-log10(p)" if not normalize_axis else "Normalized Value",
                "title_side": "right",
                "title_font": {"size": font_size + 2},
                "tickfont": {"size": font_size},
            },
        )
        return fig

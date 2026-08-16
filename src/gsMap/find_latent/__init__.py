from gsMap.logging_config import logged_stage

from .find_latent_representation import (
    run_find_latent_representation as _run_find_latent_representation,
)


@logged_stage("find_latent")
def run_find_latent_representation(config):
    """Run latent representation learning with automatic project logging."""

    return _run_find_latent_representation(config)


__all__ = ["run_find_latent_representation"]

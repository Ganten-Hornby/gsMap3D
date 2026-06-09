.. _installation:

Installation
============

**gsMap3D** requires Python 3.12 or later. We recommend using **uv** for fast and reliable package management, but Conda is also supported.

gsMap3D builds on both **PyTorch** (GNN representation learning) and **JAX** (LD score
regression). Each ships separate CPU / CUDA / TPU wheels, so the install command you choose
decides which compute backend you get. The key rule: **keep PyTorch and JAX on the same
CUDA major version** so they share a single set of NVIDIA runtime libraries.

Method 1: Use uv (Recommended)
------------------------------

`uv <https://github.com/astral-sh/uv>`_ is an extremely fast Python package installer and resolver.

1. **Install uv**:

   .. code-block:: bash

      # On Linux/macOS
      curl -LsSf https://astral.sh/uv/install.sh | sh

      # On Windows
      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

      # Or via pip
      pip install uv

2. **Install gsMap3D (CPU)**:

   .. code-block:: bash

      # Create and activate a virtual environment
      uv venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # CPU-only install (works on any machine)
      uv pip install gsMap3D --torch-backend=cpu

   This installs CPU builds of both PyTorch and JAX. For GPU acceleration, see
   `GPU Support (Optional)`_ below.

   **CLI-only install (optional)**

   If you only need the ``gsmap`` command-line tool and not the Python library:

   .. code-block:: bash

      uv tool install gsMap3D

Method 2: Use Conda
-------------------

You can also use Conda to manage your environment.

.. code-block:: bash

    # Create a new environment with Python >= 3.12
    conda create -n gsMap3D python=3.12

    # Activate the environment
    conda activate gsMap3D

    # Install gsMap3D (CPU) via pip
    pip install gsMap3D

For GPU support inside a Conda environment, install ``uv`` (``pip install uv``) and follow
`GPU Support (Optional)`_, or install the matching PyTorch build manually from the
`PyTorch index <https://download.pytorch.org/whl/>`_.

Install from Source (Developers)
--------------------------------

When you install from a clone, the repository's ``pyproject.toml`` routes PyTorch to the
matching wheel index for you, so a single ``--extra`` selects a fully harmonized
PyTorch + JAX stack:

.. code-block:: bash

    git clone https://github.com/Ganten-Hornby/gsMap3D.git
    cd gsMap3D

    uv sync --extra cpu        # CPU PyTorch + CPU JAX (use this for CI)
    uv sync --extra cuda128    # PyTorch cu128 + jax[cuda12]  (CUDA 12.8)
    uv sync --extra cuda130    # PyTorch cu130 + jax[cuda13]  (CUDA 13.0)
    uv sync --extra tpu        # JAX TPU

Pick exactly **one** backend extra — they are declared mutually exclusive. Test and docs
toolchains are dependency groups: ``uv sync --extra cpu`` already includes ``tests``; add
``--group doc`` for the documentation build.

Verification
------------

Verify the installation by checking the version:

.. code-block:: bash

    gsmap --version

GPU Support (Optional)
----------------------

**gsMap3D** uses GPU acceleration through JAX and PyTorch for significantly faster
computation.

1. **Check your CUDA / driver version**:

   .. code-block:: bash

       nvidia-smi      # top-right shows the maximum CUDA version your driver supports
       nvcc --version  # if the CUDA toolkit is installed

   CUDA 13 wheels require a recent NVIDIA driver. If unsure, choose the CUDA 12 option.

2. **Install with a matched PyTorch + JAX backend**:

   The ``cudaXXX`` extra selects JAX's CUDA build; ``--torch-backend`` selects the matching
   PyTorch wheel. Keep both on the same CUDA major.

   **CUDA 12.x:**

   .. code-block:: bash

       uv pip install "gsMap3D[cuda128]" --torch-backend=cu128

   **CUDA 13.x:**

   .. code-block:: bash

       uv pip install "gsMap3D[cuda130]" --torch-backend=cu130

   **Auto-detect PyTorch** (uv picks the Torch CUDA build matching your driver — choose the
   JAX extra for the same CUDA major):

   .. code-block:: bash

       uv pip install "gsMap3D[cuda130]" --torch-backend=auto

   .. note::

       - ``--torch-backend`` is a ``uv pip`` feature (uv >= 0.5.3) that selects the PyTorch
         wheel index. It is **not** used by ``uv sync`` or plain ``pip`` — installing from a
         clone with ``uv sync --extra cuda130`` already routes PyTorch for you.
       - With plain ``pip``, install gsMap3D and then the matching Torch build explicitly,
         e.g. ``pip install "gsMap3D[cuda130]"`` followed by
         ``pip install torch --index-url https://download.pytorch.org/whl/cu130``.
       - The JAX team recommends migrating to CUDA 13 wheels for future compatibility. See the
         `JAX Installation Guide <https://docs.jax.dev/en/latest/installation.html>`_.

3. **Verify GPU support**:

   After installation, verify that both JAX and PyTorch detect your GPU and report the same
   CUDA major version:

   **Verify JAX GPU:**

   .. code-block:: bash

       python -c "import jax; print('JAX devices:', jax.devices())"

   Expected output should show your GPU device(s), e.g., ``[CudaDevice(id=0)]``.

   **Verify PyTorch GPU:**

   .. code-block:: bash

       python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('torch CUDA:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

   Expected output should show ``CUDA available: True``, a ``torch CUDA`` version matching your
   JAX build's CUDA major (e.g. ``12.8`` with ``cuda128``, ``13.0`` with ``cuda130``), and your
   GPU name.

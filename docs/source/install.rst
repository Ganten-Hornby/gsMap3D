.. _installation:

Installation
============

**gsMap** requires Python 3.12 or later. We recommend using **uv** for fast and reliable package management, but Conda is also supported.

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

2. **Install gsMap**:

   **Option A: Using a virtual environment (Recommended for projects)**

   .. code-block:: bash

      # Create a virtual environment
      uv venv
      
      # Activate it
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate
      
      # Install gsMap
      uv pip install gsMap

   **Option B: Using uv tools (For CLI usage only)**

   If you only need the ``gsmap`` command-line tool and don't plan to use it as a library in scripts:

   .. code-block:: bash

      uv tool install gsMap

   **Option C: Install from Source with uv**

   .. code-block:: bash

      git clone https://github.com/JianYang-Lab/gsMap.git
      cd gsMap
      uv pip install .

Method 2: Use Conda
-------------------

You can also use Conda to manage your environment.

.. code-block:: bash

    # Create a new environment with Python >= 3.12
    conda create -n gsMap python=3.12
    
    # Activate the environment
    conda activate gsMap
    
    # Install gsMap via pip
    pip install gsMap

Source Installation
-------------------

To install the latest development version from GitHub:

.. code-block:: bash

    git clone https://github.com/JianYang-Lab/gsMap.git
    cd gsMap
    pip install .

Verification
------------

Verify the installation by checking the version:

.. code-block:: bash

    gsmap --version
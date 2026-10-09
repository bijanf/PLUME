PLUME
=====

|tests| |lint| |python| |ruff| |license|

.. |tests| image:: https://github.com/bijanf/PLUME/actions/workflows/tests.yml/badge.svg?branch=main
   :target: https://github.com/bijanf/PLUME/actions/workflows/tests.yml
   :alt: Tests
.. |lint| image:: https://github.com/bijanf/PLUME/actions/workflows/lint.yml/badge.svg?branch=main
   :target: https://github.com/bijanf/PLUME/actions/workflows/lint.yml
   :alt: Lint
.. |python| image:: https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white
   :target: https://www.python.org
   :alt: Python 3.12 and 3.13
.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Linted and formatted with Ruff
.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: LICENSE
   :alt: MIT licence

Code and results for *Plume flux and the resolvable features of megaplumes
triggered in the deep sea* by Masoud Rostami, Bijan Fallah, Wanting Hou and
Li-Yun Fu (manuscript under review).

Submarine eruptions release heat in bursts large enough to form megaplumes, and
the seafloor tephra they leave behind is often the only record of the
discharge. The package ``plume_inv`` inverts every sieve fraction of such a
deposit jointly for the mass-weighted distribution of umbrella volume flux, and
converts it to a heat flux through the plume-stem closure of Morton, Taylor and
Turner. It couples a quasi-steady and an unsteady, front-limited umbrella cloud
with a settling law for basaltic clasts in seawater, measures how many modes of
the source a deposit resolves, and compares candidate heat sources by Bayesian
evidence. The analysis scripts apply it to the push cores of Clague, Paduan and
Davis (2009) from the North Escanaba (NESCA) site on the southern Gorda Ridge.

.. contents:: Contents
   :local:
   :depth: 1

Quick start
-----------

Python 3.12 or newer is required.

.. code-block:: bash

   git clone https://github.com/bijanf/PLUME.git
   cd PLUME
   python -m pip install -e ".[dev]"
   python scripts/fetch_data.py      # input workbooks, about 0.4 MB, checksum verified
   python -m pytest

The exact environment that produced ``results/`` is pinned in
``env/environment.yml``. Use it on Intel Macs, for which JAX publishes no
wheels on PyPI:

.. code-block:: bash

   mamba env create -f env/environment.yml
   mamba activate plume
   python -m pip install --no-deps -e .

Reproducing the paper
---------------------

Every number quoted in the paper is read from ``results/manifest.json``, which
holds 290 values under stable dotted keys such as ``site.N_buoyancy_s``. Each
experiment writes one result file, and the manifest is assembled from them.
The result files are committed, so the figures can be drawn without
recomputing anything:

.. code-block:: bash

   make figures    # draw the six figures into figures/
   make results    # fetch all input data and recompute every file in results/
   make help       # list every target

``make results`` runs the experiments in dependency order, each one reading
only the result files written before it. The table maps every figure of the
paper to the script that draws it and the result files it reads.

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Figure
     - Script
     - Result files
   * - 1
     - ``figures/src/fig_kernels.py``
     - ``kernel_maps.json``, ``kernel_maps_grids.npz``, ``stratification.json``
   * - 2
     - ``figures/src/fig_svd.py``
     - ``identifiability.json``, ``modes.json``
   * - 3
     - ``figures/src/fig_nesca.py``
     - ``nesca_maps.json``, ``nesca_steady.json``
   * - 4
     - ``figures/src/fig_vent.py``
     - ``vent_shape.json``, ``vent_misfit.json``, ``nesca_unsteady.json``,
       ``nesca_maps.json``, ``shape_sensitivity.json``
   * - 5
     - ``figures/src/fig_shape.py``
     - ``shape_space.json``
   * - 6
     - ``figures/src/fig_sources.py``
     - ``heat_context.json``, ``model_comparison.json``,
       ``model_comparison_by_shape.json``, ``vent_shape.json``

Thinned posterior samples are stored in ``results/nesca_maps.json`` and
``results/vent_shape.json``. ``results/heat_flux_context.json`` is a curated
compilation of literature heat fluxes with the source of every value; it is an
input to ``experiments/heat_context`` and is not regenerated.

Repository layout
-----------------

.. code-block:: text

   src/plume_inv/        model and inversion package
   experiments/<name>/   one analysis per directory; run.py writes results/<name>.json
   figures/src/          scripts that draw the figures of the paper
   results/              result files and the manifest of every quoted number
   scripts/fetch_data.py downloads the input data and verifies their checksums
   data/raw/             provenance, URLs and SHA-256 checksums of the input data
   tests/                test suite
   env/environment.yml   pinned conda environment

The package modules, in the order the model is built:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Contents
   * - ``constants``
     - Physical constants and benchmark parameter values, each with its source.
   * - ``stem``
     - Plume-stem closures for point and planar sources, and the heat flux.
   * - ``settling``
     - Settling velocity of basaltic tephra in cold seawater, by clast shape.
   * - ``umbrella``
     - Umbrella-cloud dynamics, quasi-steady and front-limited.
   * - ``kernel``
     - Deposition kernels and the representation of the source as a measure
       over volume flux.
   * - ``spacetime``
     - Space-time structure of the front-limited deposit.
   * - ``forward``
     - Deposit prediction for every size class at every core.
   * - ``data``
     - Loader for the NESCA push cores, with provenance checks.
   * - ``svd``
     - Identifiability analysis: the number of source modes a deposit resolves.
   * - ``inverse``
     - Bayesian inversion with NumPyro.
   * - ``maps``
     - Map-view deposit and the misfit over trial vent positions.
   * - ``shape``
     - Clast shape from the spread of dispersal length scales.
   * - ``sources``
     - Candidate heat sources for a megaplume, with physically bounded parameters.
   * - ``heat_context``
     - The megaplume heat flux set against other geophysical heat fluxes.

Data
----

The input data are fetched from their original providers and checked against
the SHA-256 checksums recorded in ``scripts/fetch_data.py`` and in the README
of each ``data/raw/`` directory. They are not redistributed here.

NESCA push cores
   Appendix A of Clague, D. A., Paduan, J. B. & Davis, A. S. (2009), Widespread
   strombolian eruptions of mid-ocean ridge basalt, *Journal of Volcanology and
   Geothermal Research* 180, 171–188, `doi:10.1016/j.jvolgeores.2008.08.007
   <https://doi.org/10.1016/j.jvolgeores.2008.08.007>`_. Fetched with
   ``python scripts/fetch_data.py``.

World Ocean Atlas 2023
   Annual mean temperature and salinity on the 1° grid from NOAA NCEI, in the
   public domain. Needed only by ``experiments/stratification``. Fetched with
   ``python scripts/fetch_data.py --woa`` (about 162 MB).

Development
-----------

.. code-block:: bash

   make test             # run the test suite
   make lint             # ruff check and ruff format --check
   pre-commit install    # run the same checks before every commit

Tests marked ``nesca_data`` read the NESCA workbooks and are skipped, with a
pointer to ``scripts/fetch_data.py``, when the workbooks are absent. Continuous
integration fetches the workbooks, sets ``PLUME_REQUIRE_DATA=1`` so that a
missing file fails the run, and tests on Linux with Python 3.12 and 3.13 and on
macOS with Python 3.12. Each run reports the line and branch coverage of
``plume_inv`` in its summary and fails below 80 %. Bug reports and questions are welcome as
`GitHub issues <https://github.com/bijanf/PLUME/issues>`_.

Citation
--------

If you use this code, please cite the paper. GitHub's *Cite this repository*
button reads the metadata in ``CITATION.cff``.

   Rostami, M., Fallah, B., Hou, W. & Fu, L.-Y. Plume flux and the resolvable
   features of megaplumes triggered in the deep sea. Manuscript under review
   (2026).

License
-------

The code is released under the MIT License; see ``LICENSE``. The input data
remain under the terms of their providers.

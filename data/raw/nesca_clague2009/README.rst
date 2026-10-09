NESCA push-core tephra data: Clague, Paduan & Davis (2009), Appendix A
======================================================================

The workbooks are **not** part of this repository.  They are the publisher's
supplementary files and are redistributed by reference.  Fetch and verify them
with::

    python scripts/fetch_data.py

Nothing in this directory is edited.  The loader ``plume_inv.data.load_nesca``
reads the workbooks as published and checks them on every load.

Source
------
The push-core tephra dataset inverted by Pegler & Ferguson (2021), whose data
availability statement points to:

    Clague, D. A., Paduan, J. B. & Davis, A. S. (2009).  Widespread strombolian
    eruptions of mid-ocean ridge basalt.  *Journal of Volcanology and
    Geothermal Research* **180**, 171-188.  doi:10.1016/j.jvolgeores.2008.08.007

The Appendix A supplementary workbooks are served openly by Elsevier's CDN under
the article PII ``S0377027308004563`` (from the Crossref record of the DOI
above).  The files were downloaded on 2026-09-16; two independent downloads
were byte-identical.

Files
-----
.. list-table::
   :header-rows: 1

   * - File
     - URL
     - Bytes
     - SHA-256
   * - ``mmc1.xls``
     - https://ars.els-cdn.com/content/image/1-s2.0-S0377027308004563-mmc1.xls
     - 135168
     - ``8583f42bba0160ce5e8c48303b4c63476768bc8ae63d9a704074b3ca74d6ee6e``
   * - ``mmc2.xls``
     - https://ars.els-cdn.com/content/image/1-s2.0-S0377027308004563-mmc2.xls
     - 216576
     - ``4194e8768dbad9a3185e23fd13282e2e3a172348b073e0db8e100d416a25a4bd``
   * - ``mmc3.xls``
     - https://ars.els-cdn.com/content/image/1-s2.0-S0377027308004563-mmc3.xls
     - 71680
     - ``5dd7e52fdecf8d70f860403d7cd472e20dde162ef5a1aee8899c5f05425ed606``

The CDN rejects requests without a browser-like User-Agent.  To check by hand::

    for n in 1 2 3; do
      curl -sL -A "Mozilla/5.0" \
        "https://ars.els-cdn.com/content/image/1-s2.0-S0377027308004563-mmc${n}.xls" \
        | shasum -a 256
    done

Contents
--------
``mmc3.xls``, sheet ``Sheet1``: the tephra data (132 cores)
    Columns ``Sample``, ``Location`` (seafloor character), ``Largest Three Chips
    (g)`` (3 columns), then four sieve fractions each as a ``(g, %Glass)`` pair,
    **63-125 um, 125-250 um, 250-500 um, >500 um**, then ``Total Glass (g)`` and
    ``g/m2``.  132 data rows plus five footnote rows.  The footnotes state that
    the fractions were wet sieved, dried overnight at 80 degC, weighed, and the
    percent glass visually estimated; that the upper 8 cm of each core were
    processed (2 cm increments, combined, for samples marked ``**``); that
    0.1 % was used for ``<<1`` and 0.3 % for ``<1``; and that ``*`` marks
    samples where some 63-125 um fraction was lost.

``mmc1.xls``, sheet ``LimuSamples``: sample coordinates (467 samples)
    Columns ``Sample``, ``Waypoint``, ``Equipment``, ``Latitude``,
    ``Longitude``, ``Depth``, ``Dive``, ``Date_Time``, ``TapeTimeCode``,
    ``Concept``, ``Comment``.  All 132 ``mmc3`` cores join by ``Sample``, all
    with ``Equipment = push-corer``, across ROV dives T887-T891: latitude
    40.9362-41.0565 N, longitude 127.5436-127.4278 W, water depth 3212-3371 m.

``mmc2.xls``
    Glass major-element compositions.  Not used by the analysis.

Two properties of the data the loader handles
---------------------------------------------
1. **Two core areas.**  The footnote quotes one core diameter (6.9 cm), but
   dividing ``Total Glass (g)`` by ``g/m2`` gives exactly two areas:
   3.739e-3 m^2 (6.90 cm; 103 cores, dives T887, T889, T890, T891) and
   3.939e-3 m^2 (7.08 cm; 29 cores, dive T888 only).  Any re-derivation of mass
   per unit area must apply the per-dive area, or it introduces a 5.3 % step
   between dive T888 and the rest.  The loader recovers the area of every core
   from the published columns and refuses the data if they imply any other
   area, or if the larger area is not confined to dive T888.
2. **Completeness.**  131 of 132 cores have numeric masses in all four
   fractions; one core has none, and one carries the string ``lost`` in the
   63-125 um column.  A missing fraction is treated as missing, not as zero.

Licence
-------
Elsevier supplementary material accompanying a subscription article, downloaded
from the publisher's open CDN for research use.  Do not redistribute the
workbooks; cite Clague, Paduan & Davis (2009) when using them.

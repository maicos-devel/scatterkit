Scatterkit
==========

|tests| |codecov| |docs-stable| |docs-latest| |mdanalysis|

.. inclusion-readme-intro-start

**Scatterkit** is an object-oriented python toolkit for analysing scattering of
fluids from molecular simulations. Combined with MDAnalysis_, 
Scatterkit can be used to extract X-ray scattering data from trajectory files, 
including LAMMPS, GROMACS, CHARMM or NAMD data. Scatterkit is open source and is
released under the GNU general public license v3.0.

Scatterkit is a tool for beginners of molecular simulations with no prior Python experience.
For these users Scatterkit provides a descriptive command line interface. Also experienced
users can use the Python API for their day to day analysis. 

Scatterkit is maintained by the MAICoS developer team.
Keep up to date with Scatterkit news by following us on Twitter_. If you find an issue, you
can report it on GitHub_. You can also join the developer team on Discord_ to discuss
possible improvements and usages of Scatterkit.

.. _`MDAnalysis`: https://www.mdanalysis.org
.. _`Twitter`: https://twitter.com/maicos_analysis
.. _`GitHub`: https://github.com/maicos-devel/scatterkit
.. _`Discord`: https://discord.gg/mnrEQWVAed

.. inclusion-readme-intro-end

Documentation
=============

For details, tutorials, and examples, visit our official `documentation`_. We also
provide the `latest documentation`_ for the current development version of Scatterkit.

.. _`documentation`: https://maicos-devel.github.io/scatterkit
.. _`latest documentation`: https://maicos-devel.github.io/scatterkit/latest

.. inclusion-readme-installation-start

Installation
============

Install Scatterkit using `pip`_::

    pip install scatterkit

Or with `conda`_ from conda-forge::

    conda install -c conda-forge scatterkit

.. _`pip`: https://pip.pypa.io
.. _`conda`: https://www.anaconda.com

.. inclusion-readme-installation-end

List of Analysis Modules
========================

.. inclusion-marker-modules-start

Currently, Scatterkit supports the following analysis modules (alphabetically):

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Module
     - Description
   * - SAXS
     - Small angle X-ray scattering

.. inclusion-marker-modules-end

Contributors
============

Thanks to all contributors who make **Scatterkit** possible:

.. image:: https://contrib.rocks/image?repo=maicos-devel/scatterkit
   :target: https://github.com/maicos-devel/scatterkit/graphs/contributors

.. |tests| image:: https://github.com/maicos-devel/scatterkit/workflows/Tests/badge.svg
   :alt: GitHub Actions Tests Job Status
   :target: https://github.com/maicos-devel/scatterkit/actions?query=branch%3Amain

.. |codecov| image:: https://codecov.io/gh/maicos-devel/scatterkit/graph/badge.svg?token=9AXPLF6CR3
   :alt: Code coverage
   :target: https://codecov.io/gh/maicos-devel/scatterkit

.. |docs-stable| image:: https://img.shields.io/badge/📚_Documentation-stable-success
   :alt: Documentation of stable released version
   :target: `documentation`_

.. |docs-latest| image:: https://img.shields.io/badge/📒_Documentation-latest-yellow.svg
   :alt: Documentation of latest unreleased version
   :target: `latest documentation`_

.. |mdanalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org

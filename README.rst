====================
Differentially Private Mobility Data Report
====================


.. image:: https://img.shields.io/pypi/v/dp_mobility_report.svg
        :target: https://pypi.python.org/pypi/dp_mobility_report

.. image:: https://img.shields.io/travis/AlexandraKapp/dp_mobility_report.svg
        :target: https://travis-ci.com/AlexandraKapp/dp_mobility_report

.. image:: https://readthedocs.org/projects/mobility-data-report/badge/?version=latest
        :target: https://mobility-data-report.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



!!! THIS PACKAGE IS STILL UNDER CONSTRUCTION  !!!

* Free software: MIT license
* Documentation: https://mobility-data-report.readthedocs.io.


``dp_mobility_report``: A python package to create a standardized mobility report with differential privacy guarentees, especially for urban human mobility data.


Install
**********************

.. code-block:: bash

        pip install git+https://github.com/FreeMoveProject/dp_mobility_report

Create a mobility report as HTML:
**********************

.. code-block:: python

        import pandas as pd
        import geopandas as gpd
        from dp_mobility_report import md_report

        # -- insert paths --
        df = pd.read_csv("mobility_dataset.csv")
        tessellation = gpd.read_file("tessellation.gpkg")

        report = md_report.MobilityDataReport(
            df, tessellation, privacy_budget=1, max_trips_per_user=4
        )

        report.to_file("my_mobility_report.html")


Example HTMLs can be found in the examples_ folder.


Credits
-------

This package was highly inspired by the `pandas-profiling/pandas-profiling`_ and `scikit-mobility`_ packages.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
 
.. _examples: https://github.com/FreeMoveProject/dp_mobility_report/tree/main/examples/html
.. _`pandas-profiling/pandas-profiling`: https://github.com/pandas-profiling/pandas-profiling
.. _`scikit-mobility`: https://github.com/scikit-mobility
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

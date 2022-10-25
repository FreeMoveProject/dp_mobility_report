============================================================
Differentially Private Mobility Data Report
============================================================


.. image:: https://img.shields.io/pypi/v/dp_mobility_report.svg
        :target: https://pypi.python.org/pypi/dp_mobility_report

        
.. image:: https://readthedocs.org/projects/dp-mobility-report/badge/?version=latest
        :target: https://dp-mobility-report.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




* Free software: MIT license
* Documentation: https://dp-mobility-report.readthedocs.io.


``dp_mobility_report``: A python package to create a mobility report with differential privacy guarentees, especially for urban human mobility data. 


Install
**********************

.. code-block:: bash

        pip install dp-mobility-report

or from GitHub:

.. code-block:: bash

        pip install git+https://github.com/FreeMoveProject/dp_mobility_report


Data preparation
**********************

- **df**: a pandas ``DataFrame``. Expected columns: User ID ``uid``, Trip ID ``tid``, timestamp `datetime` (or `int`to indicate sequence position, if dataset only consists of sequences without timestamps), latitude and longitude in CRS EPSG:4326 ``lat`` and ``lng``.
- **tessellation**: a geopandas ``GeoDataFrame`` with polygons. Expected columns ``tile_id``. The tessellation is used for spatial aggregations of the data.

Create a mobility report as HTML 
**************************************

.. code-block:: python

        import pandas as pd
        import geopandas as gpd
        from dp_mobility_report import DpMobilityReport

        # -- insert paths --
        df = pd.read_csv("mobility_dataset.csv")
        tessellation = gpd.read_file("tessellation.gpkg")

        report = DpMobilityReport(df, tessellation, privacy_budget=1, max_trips_per_user=4)

        report.to_file("my_mobility_report.html")


The parameter ``privacy_budget`` (in terms of *epsilon*-differential privacy) determines how much noise is added to the data. The budget is split between all analyses of the report.
If the value is set to ``None`` no noise (i.e., no privacy guarantee) is applied to the report.

The parameter ``max_trips_per_user`` specifies how many trips a user can contribute to the dataset at most. If a user is represented with more trips, a random sample is drawn according to ``max_trips_per_user``.
If the value is set to ``None`` the full dataset is used. Note, that deriving the maximum trips per user from the data violates the differential privacy guarantee. Thus, ``None`` should only be used in combination with ``privacy_budget=None``.

Please refer to the `documentation`_ for information on further parameters.

Example HTMLs can be found in the examples_ folder.


Credits
-------

This package was highly inspired by the `pandas-profiling/pandas-profiling`_ and `scikit-mobility`_ packages.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
 
.. _documentation: https://dp-mobility-report.readthedocs.io/en/latest/modules.html
.. _examples: https://github.com/FreeMoveProject/dp_mobility_report/tree/main/examples/html
.. _`pandas-profiling/pandas-profiling`: https://github.com/pandas-profiling/pandas-profiling
.. _`scikit-mobility`: https://github.com/scikit-mobility
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

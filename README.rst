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


``dp_mobility_report``: A python package to create a mobility report with differential privacy (DP) guarentees, especially for urban human mobility data. 


Install
**********************

.. code-block:: bash

        pip install dp-mobility-report

or from GitHub:

.. code-block:: bash

        pip install git+https://github.com/FreeMoveProject/dp_mobility_report


Data preparation
**********************

**df**: 

* A pandas ``DataFrame``. 
* Expected columns: User ID ``uid``, Trip ID ``tid``, timestamp ``datetime`` (expected is a datetime-like string, e.g., in the format ``yyyy-mm-dd hh:mm:ss``. If ``datetime`` contains ``int`` values, it is interpreted as sequence positions, i.e., if the dataset only consists of sequences without timestamps), latitude and longitude in CRS EPSG:4326 ``lat`` and ``lng``. 
* Here you can find an `example dataset`_.

**tessellation**: 

* A geopandas ``GeoDataFrame`` with polygons. 
* Expected columns: ``tile_id``. 
* The tessellation is used for spatial aggregations of the data. 
* Here you can find an `example tessellation`_. 
* If you don't have a tessellation, you can use this code to `create a tessellation`_.

Create a mobility report as HTML
**************************************

.. code-block:: python

        import pandas as pd
        import geopandas as gpd
        from dp_mobility_report import DpMobilityReport

        df = pd.read_csv("https://raw.githubusercontent.com/FreeMoveProject/dp_mobility_report/main/tests/test_files/test_data.csv")
        tessellation = gpd.read_file("https://raw.githubusercontent.com/FreeMoveProject/dp_mobility_report/main/tests/test_files/test_tessellation.geojson")

        report = DpMobilityReport(df, tessellation, privacy_budget=10, max_trips_per_user=5)

        report.to_file("my_mobility_report.html")


The parameter ``privacy_budget`` (in terms of *epsilon*-DP) determines how much noise is added to the data. The budget is split between all analyses of the report.
If the value is set to ``None`` no noise (i.e., no privacy guarantee) is applied to the report.

The parameter ``max_trips_per_user`` specifies how many trips a user can contribute to the dataset at most. If a user is represented with more trips, a random sample is drawn according to ``max_trips_per_user``.
If the value is set to ``None`` the full dataset is used. Note, that deriving the maximum trips per user from the data violates the differential privacy guarantee. Thus, ``None`` should only be used in combination with ``privacy_budget=None``.

Please refer to the `documentation`_ for information on further parameters.


Examples
*********

Berlin mobility data simulated using the `DLR TAPAS`_ Model: [`Code used for Berlin`_]

* `Report of Berlin without DP`_
* `Report of Berlin with DP epsilon=1`_

Madrid `CRTM survey`_ data: [`Code used for Madrid`_]

* `Report of Madrid without DP`_
* `Report of Madrid with DP epsilon=10`_

Beijing `Geolife`_ dataset: [`Code used for Beijing`_]

* `Report of Beijing without DP`_
* `Report of Beijing with DP epsilon=50`_

(Here you find the `code of the data preprocessing`_ to obtain the needed format)

Credits
-------

This package was highly inspired by the `pandas-profiling/pandas-profiling`_ and `scikit-mobility`_ packages.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
 
.. _`example dataset`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/tests/test_files/test_data.csv
.. _`example tessellation`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/tests/test_files/test_tessellation.geojson
.. _`create a tessellation`:  https://github.com/FreeMoveProject/dp_mobility_report/blob/main/examples/create_tessellation.py
.. _documentation: https://dp-mobility-report.readthedocs.io/en/latest/modules.html
.. _examples: https://github.com/FreeMoveProject/dp_mobility_report/tree/main/examples/html
.. _`pandas-profiling/pandas-profiling`: https://github.com/pandas-profiling/pandas-profiling
.. _`scikit-mobility`: https://github.com/scikit-mobility
.. _`DLR TAPAS`: https://github.com/DLR-VF/TAPAS
.. _`Report of Berlin without DP`: https://freemoveproject.github.io/dp_mobility_report/examples/html/berlin_noPrivacy.html
.. _`Report of Berlin with DP epsilon=1`: https://freemoveproject.github.io/dp_mobility_report/examples/html/berlin.html
.. _`Code used for Berlin`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/examples/example_berlin.py
.. _`CRTM survey`: https://crtm.maps.arcgis.com/apps/MinimalGallery/index.html?appid=a60bb2f0142b440eadee1a69a11693fc
.. _`Report of Madrid without DP`: https://freemoveproject.github.io/dp_mobility_report/examples/html/madrid_noPrivacy.html
.. _`Report of Madrid with DP epsilon=10`: https://freemoveproject.github.io/dp_mobility_report/examples/html/madrid.html
.. _`Code used for Madrid`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/examples/example_madrid.py
.. _`Geolife`: https://www.microsoft.com/en-us/download/details.aspx?id=52367
.. _`Report of Beijing without DP`: https://freemoveproject.github.io/dp_mobility_report/examples/html/geolife_noPrivacy.html
.. _`Report of Beijing with DP epsilon=50`: https://freemoveproject.github.io/dp_mobility_report/examples/html/geolife.html
.. _`Code used for Beijing`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/examples/example_geolife.py
.. _`code of the data preprocessing`: https://github.com/FreeMoveProject/evaluation_dp_mobility_report/blob/main/01_preprocess_evaluation_data.py
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

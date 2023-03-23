============================================================
Differentially Private Mobility Report (DpMobilityReport)
============================================================


.. image:: https://img.shields.io/pypi/v/dp_mobility_report.svg
        :target: https://pypi.python.org/pypi/dp_mobility_report

        
.. image:: https://readthedocs.org/projects/dp-mobility-report/badge/?version=latest
        :target: https://dp-mobility-report.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




* Free software: MIT license
* Documentation: https://dp-mobility-report.readthedocs.io.


``dp_mobility_report``: A python package to create a mobility report with differential privacy (DP) guarantees, especially for urban human mobility data. 


Quickstart 
**************

Install
==========

.. code-block:: bash

        pip install dp-mobility-report

or from GitHub:

.. code-block:: bash

        pip install git+https://github.com/FreeMoveProject/dp_mobility_report


Data preparation
====================

**df**: 

* A pandas ``DataFrame``. 
* Expected columns: User ID ``uid``, Trip ID ``tid``, timestamp ``datetime`` (expected is a datetime-like string, e.g., in the format ``yyyy-mm-dd hh:mm:ss``. If ``datetime`` contains ``int`` values, it is interpreted as sequence positions, i.e., if the dataset only consists of sequences without timestamps), latitude and longitude in CRS EPSG:4326 ``lat`` and ``lng``. (We thereby closely followed the format of the `scikit-mobility`_ ``TrajDataFrame``.)
* Here you can find an `example dataset`_.

**tessellation**: 

* A geopandas ``GeoDataFrame`` with polygons. 
* Expected columns: ``tile_id``. 
* The tessellation is used for spatial aggregations of the data. 
* Here you can find an `example tessellation`_. 
* If you don't have a tessellation, you can use this code to `create a tessellation`_.


Create a DpMobilityReport
===================================

.. code-block:: python

        import pandas as pd
        import geopandas as gpd
        from dp_mobility_report import DpMobilityReport

        df = pd.read_csv(
            "https://raw.githubusercontent.com/FreeMoveProject/dp_mobility_report/main/tests/test_files/test_data.csv"
        )
        tessellation = gpd.read_file(
            "https://raw.githubusercontent.com/FreeMoveProject/dp_mobility_report/main/tests/test_files/test_tessellation.geojson"
        )

        report = DpMobilityReport(df, tessellation, privacy_budget=10, max_trips_per_user=5)

        report.to_file("my_mobility_report.html")


The parameter ``privacy_budget`` (in terms of *epsilon*-DP) determines how much noise is added to the data. The budget is split between all analyses of the report.
If the value is set to ``None`` no noise (i.e., no privacy guarantee) is applied to the report.

The parameter ``max_trips_per_user`` specifies how many trips a user can contribute to the dataset at most. If a user is represented with more trips, a random sample is drawn according to ``max_trips_per_user``.
If the value is set to ``None`` the full dataset is used. Note, that deriving the maximum trips per user from the data violates the differential privacy guarantee. Thus, ``None`` should only be used in combination with ``privacy_budget=None``.

Please refer to the `documentation`_ for information on further parameters. Here you can find information on the `analyses`_ of the report.

Example HTMLs can be found in the examples_ folder.


Create a BenchmarkReport 
================================

A benchmark report evaluate the similarity of two (differentially private) mobility reports from one or two mobility datasets. This can be based on two datasets (``df_base`` and ``df_alternative``) or one dataset (``df_base``)) with different privacy settings.
The arguments ``df``, ``privacy_budget``, ``user_privacy``, ``max_trips_per_user`` and ``budget_split`` can differ for the two datasets set with the according ending ``_base`` and ``_alternative``. The other arguments are the same for both reports.
For the evaluation, `similarity measures`_ (namely the (mean) absolute percentage error (PE), Jensen-Shannon divergence (JSD), Kullback-Leibler divergence (KLD), and the earth mover's distance (EMD)) are computed to quantify the statistical similarity for each analysis.
The evaluation, i.e., benchmark report, will be generated as an HTML file, using the ``.to_file()`` method.


Benchmark of two different datasets 
---------------------------------------

This example creates a benchmark report with similarity measures for two mobility datasets, called *base* and *alternative* in the following. This is intended to compare different datasets with the same or no privacy budget.

.. code-block:: python

        import pandas as pd
        import geopandas as gpd
        from dp_mobility_report import BenchmarkReport

        # -- insert paths --
        df_base = pd.read_csv("mobility_dataset_base.csv")
        df_alternative = pd.read_csv("mobility_dataset_alternative.csv")
        tessellation = gpd.read_file("tessellation.gpkg")

        benchmark_report = BenchmarkReport(
            df_base=df_base, tesselation=tessellation, df_alternative=df_alternative
        )

        # Dictionary containing the similarity measures for each analysis
        similarity_measures = benchmark_report.similarity_measures
        # The measure selection indicates which similarity measure
        # (e.g. KLD, JSD, EMD, PE) has been selected for each analysis
        measure_selection = benchmark_report.measure_selection

        # If you do not want to access the selection of similarity measures
        # but e.g. the Jensen-Shannon divergence for all analyses:
        jsd = benchmark_report.jsd

        # benchmark_report.to_file("my_benchmark_mobility_report.html")


The parameter ``measure_selection`` specifies which similarity measures should be chosen for the ``similarity_measures`` dictionary that is an attribute of the ``BenchmarkReport``. 
The default is set to a specific set of similarity measures for each analysis which can be accessed by ``dp_mobility_report.default_measure_selection()``. 
The default of single analyses can be overwritten as shown in the following:

.. code-block:: python

        from dp_mobility_report import BenchmarkReport, default_measure_selection
        from dp_mobility_report import constants as const

        # print the default measure selection
        print(default_measure_selection())

        # change default of EMD for visits_per_tile to JSD.
        # For the other analyses the default measure is remained
        custom_measure_selection = {const.VISITS_PER_TILE: const.JSD}

        benchmark_report = BenchmarkReport(
            df_base=df_base,
            tesselation=tessellation,
            df_alternative=df_alternative,
            measure_selection=custom_measure_selection,
        )



Benchmark of the same dataset with different privacy settings
-------------------------------------------------------------------

This example creates a BenchmarkReport with similarity measures for the same mobility dataset with different privacy settings (``privacy_budget``, ``user_privacy``, ``max_trips_per_user`` and ``budget_split``) to assess the utility loss of the privacy budget for the different analyses. 

.. code-block:: python

        import pandas as pd
        import geopandas as gpd
        from dp_mobility_report import BenchmarkReport

        # -- insert paths --
        df_base = pd.read_csv("mobility_dataset_base.csv")
        tessellation = gpd.read_file("tessellation.gpkg")

        benchmark_report = BenchmarkReport(
            df_base=df_base,
            tesselation=tessellation,
            privacy_budget_base=None,
            privacy_budget_alternative=5,
            max_trips_per_user_base=None,
            max_trips_per_user_alternative=4,
        )

        similarity_measures = benchmark_report.similarity_measures

        # benchmark_report.to_file("my_benchmark_mobility_report.html")



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

Benchmark Report: [`Code used for Benchmarkreport of Berlin`_]

* `Benchmarkreport of Berlin without DP and with DP epsilon=1`_

(Here you find the `code of the data preprocessing`_ to obtain the needed format)

Citing
******
if you use dp-mobility-report please cite the `following paper`_:

.. code-block::

        @article{
                doi:10.1080/17489725.2022.2148008,
                title = {Towards Mobility Reports with User-Level Privacy},
                author = {Kapp, Alexandra and {von Voigt}, Saskia Nu{\~n}ez and Mihaljevi{\'c}, Helena and Tschorsch, Florian},
                year = {2022},
                journal = {Journal of Location Based Services},
                eprint = {https://www.tandfonline.com/doi/pdf/10.1080/17489725.2022.2148008},
                publisher = {{Taylor \& Francis}},
                doi = {10.1080/17489725.2022.2148008}
        }


Credits
========

This package was highly inspired by the `pandas-profiling/pandas-profiling`_ and `scikit-mobility`_ packages.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
 
.. _`example dataset`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/tests/test_files/test_data.csv
.. _`example tessellation`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/tests/test_files/test_tessellation.geojson
.. _`create a tessellation`:  https://github.com/FreeMoveProject/dp_mobility_report/blob/main/examples/create_tessellation.py
.. _documentation: https://dp-mobility-report.readthedocs.io/en/latest/modules.html
.. _analyses: https://dp-mobility-report.readthedocs.io/en/latest/analyses.html
.. _`similarity measures`: https://dp-mobility-report.readthedocs.io/en/latest/similarity_measures.html
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
.. _`Benchmarkreport of Berlin without DP and with DP epsilon=1`: https://freemoveproject.github.io/dp_mobility_report/examples/html/berlin_benchmark.html
.. _`Code used for Benchmarkreport of Berlin`: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/examples/example_benchmark.py
.. _`code of the data preprocessing`: https://github.com/FreeMoveProject/evaluation_dp_mobility_report/blob/main/01_preprocess_evaluation_data.py
.. _`following paper`: https://www.tandfonline.com/doi/full/10.1080/17489725.2022.2148008
.. _`pandas-profiling/pandas-profiling`: https://github.com/pandas-profiling/pandas-profiling
.. _`scikit-mobility`: https://github.com/scikit-mobility
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

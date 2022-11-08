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

- **df**: a pandas ``DataFrame``. Expected columns: User ID ``uid``, Trip ID ``tid``, timestamp ``datetime`` (expected is a datetime-like string, e.g., in the format ``yyyy-mm-dd hh:mm:ss``. If ``datetime`` contains ``int`` values, it is interpreted as sequence positions, i.e., if the dataset only consists of sequences without timestamps), latitude and longitude in CRS EPSG:4326 ``lat`` and ``lng``. You can find an example dataset `here`_.

- **tessellation**: a geopandas ``GeoDataFrame`` with polygons. Expected columns: ``tile_id``. The tessellation is used for spatial aggregations of the data.

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


Benchmark Report 
***********************

A benchmark report evaluate the similarity of two (differentially private) mobility reports from one or two mobility datasets. This can be based on two datasets (``df_base`` and ``df_alternative``) or one dataset (``df_base``)) with different privacy settings.
The arguments ``df``, ``privacy_budget``, ``user_privacy``, ``max_trips_per_user`` and ``budget_split`` can differ for the two datasets set with the according ending ``_base`` and ``_alternative``. The other arguments are the same for both reports.
For the evaluation, similarity measures (namely the relative error (RE), Jensen-Shannon divergence (JSD), Kullback-Leibler divergence (KLD), symmetric mean absolute percentage error (SMAPE) and the earth mover's distance (EMD)) are computed to quantify the statistical similarity for each analysis.
The evaluation, i.e., benchmark report, will be generated as an HTML file, using the ``.to_file()`` method.


Benchmark of two different datasets 
=============================================

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
        # (e.g. KLD, JSD, EMD, RE, SMAPE) has been selected for each analysis
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
===============================================================

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



Credits
-------

This package was highly inspired by the `pandas-profiling/pandas-profiling`_ and `scikit-mobility`_ packages.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
 
.. _here: https://github.com/FreeMoveProject/dp_mobility_report/blob/main/tests/test_files/test_data.csv
.. _documentation: https://dp-mobility-report.readthedocs.io/en/latest/modules.html
.. _examples: https://github.com/FreeMoveProject/dp_mobility_report/tree/main/examples/html
.. _`pandas-profiling/pandas-profiling`: https://github.com/pandas-profiling/pandas-profiling
.. _`scikit-mobility`: https://github.com/scikit-mobility
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

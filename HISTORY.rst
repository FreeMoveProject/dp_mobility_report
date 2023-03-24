History
*********

0.2.5 (2023-03-24)
==================
* Bug fix: compatibility with pandas >= 2.0 and pandas < 2.0

0.2.4 (2023-03-23)
==================
* Enhance HTML design 
* Include info texts for all analyses
* Include documentation for differential privacy and an info box about DP in the report
* Enhance documentation
* Add option for `subtitle` in DpMobilityReport and BenchmarkReport to name the report.

0.2.3 (2023-02-13)
==================
* Bug fix: handle if no visit is within the tessallation
* Bug fix: handle if no OD trip is within the tessallation
* Bug fix: unify histogram bins rounding issue

0.2.2 (2023-02-01)
==================
* Bug fix: exclude user_time_delta if there is no user with at least two trips.
* Bug fix: set max_trips_per_user correctly if user_privacy=False.
* Enhancement: do not exclude jump_length and travel_time if no tessellation is given

0.2.1 (2023-01-24)
==================
* Bug fix: Correct range of scale for visits per time and tile map. 

0.2.0 (2023-01-23)
==================
* Create a BenchmarkReport class that evaluates the similarity of two (differentially private) mobility reports from one or two mobility datasets and creates an HTML output similar to the DpMobilityReport.

0.1.8 (2023-01-16)
==================
* Refine handling of OD Analysis input data:
    * warn if there are no trips with more than a single record and exclude OD Analysis
    * use all trips for travel time and jump length computation instead of only trips inside tessellation.

0.1.7 (2023-01-10)
==================
* Restructuring of HTML headlines.

0.1.6 (2023-01-09)
==================
* Refactoring of template files.

0.1.5 (2022-12-12)
==================
* Remove scikit-mobility dependency and refactor od flow visualization.

0.1.4 (2022=12=07)
==================
* Remove Google Fonts from HTML.

0.1.3 (2022-12-05)
==================
* Handle FutureWarning of pandas.

0.1.2 (2022-11-24)
==================
* Enhanced documentation for all properties of `DpMobilityReport` class

0.1.1 (2022-10-27)
==================
* fix bug: prevent error "key `trips` not found" in `trips_over_time` if sum of `trip_count` is 0

0.1.0 (2022-10-21)
==================
* make tessellation an Optional parameter
* allow DataFrames without timestamps but sequence numbering instead (i.e., `integer` for `timestamp` column)
* allow to set seed for reproducible sampling of the dataset (according to `max_trips_per_user`)

0.0.8 (2022-10-20)
==================
* Fixes addressing deprecation warnings.

0.0.7 (2022-10-17)
==================

* parameter for a custom split of the privacy budget between different analyses
* extend 'analysis_selection' to include single analyses instead of entire segments
* parameter for 'analysis_exclusion' instead of selection
* bug fix: include all possible categories for days and hour of days
* bug fix: show correct percentage of outliers
* show 95% confidence-interval instead of upper and lower bound
* show privacy budget and confidence interval for each analysis

0.0.6 (2022-09-30)
==================

* Remove scaling of counts to match a consistent trip_count / record_count (from ds_statistics) in visits_per_tile, visits_per_time_tile and od_flows. Scaling was implemented to keep the report consistent, though it is removed for now as it introduces new issues.
* Minor bug fixes in the visualization: outliers were not correctly converted into percentage. 

0.0.5 (2022-08-26)
==================

Bug fix: correct scaling of timewindow counts.

0.0.4 (2022-08-22)
==================

* Simplify naming: from :code:`MobilityDataReport` to :code:`DpMobilityReport`
* Simplify import: from :code:`from dp_mobility_report import md_report.MobilityDataReport` to :code:`from dp_mobility_report import DpMobilityReport`
* Enhance documentation: change style and correctly include API reference.

0.0.3 (2022-07-22)
==================

* Fix broken link.

0.0.2 (2022-07-22)
==================

* First release to PyPi.
* It includes all basic functionality, though still in alpha version and under development.

0.0.1 (2021-12-16)
==================

* First version used for evaluation in Alexandra Kapp, Saskia Nuñez von Voigt, Helena Mihaljević & Florian Tschorsch (2022) Towards mobility reports with user-level privacy, Journal of Location Based Services, DOI: 10.1080/17489725.2022.2148008.
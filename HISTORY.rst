History
*********

0.0.6 (2022-09-30)
------------------

* Remove scaling of counts to match a consistent trip_count / record_count (from ds_statistics) in visits_per_tile, visits_per_tile_timewindow and od_flows. Scaling was implemented to keep the report consistent, though it is removed for now as it introduces new issues.
* Minor bug fixes in the visualization: outliers were not correctly converted into percentage. 

0.0.5 (2022-08-26)
------------------

Bug fix: correct scaling of timewindow counts.

0.0.4 (2022-08-22)
------------------

* Simplify naming: from :code:`MobilityDataReport` to :code:`DpMobilityReport`
* Simplify import: from :code:`from dp_mobility_report import md_report.MobilityDataReport` to :code:`from dp_mobility_report import DpMobilityReport`
* Enhance documentation: change style and correctly include API reference.

0.0.3 (2022-07-22)
------------------

* Fix broken link.

0.0.2 (2022-07-22)
------------------

* First release to PyPi.
* It includes all basic functionality, though still in alpha version and under development.

0.0.1 (2021-12-16)
------------------

* First version used for evaluation in xx.
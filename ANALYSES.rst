============================================================
Analyses in the DpMobilityReport
============================================================

In the following, all available analyses within the DP Mobility Report are explained. They are grouped by the four segments of the report.
In brackets behind the name, the variable name is stated as used in the :code:`dp_mobility_report.constants` file.

In the following, we refer to the *five-number summary* which describes a set of descriptive statistics that provides information about a dataset. It consists of the five most important sample percentiles: 
the sample minimum, the lower quartile, the median, the upper quartile and the sample maximum.

Overview
*********

* **Dataset Statistics** (:code:`DS_STATISTICS`): Number of records, number of trips, number of complete trips, number of incomplete trips, number of users, number of locations

* **Missing values** (:code:`MISSING_VALUES`): Number of missing values for user ID, trip ID, datetime, latitude, longitude
	
* **Trips over time** (:code:`TRIPS_OVER_TIME`): Based on the time granularity of the data, the relative number of trips is computed for day, week or month segments.

* **Trips per weekday** (:code:`TRIPS_PER_WEEKDAY`): Compute the relative number of trips per weekday.

* **Trips per hour** (:code:`TRIPS_PER_HOUR`): Compute the relative number of trips per hour of the day.


Place Analysis
**************

* **Visits per tile** (:code:`VISITS_PER_TILE`): Compute number of times each tile of the chosen tessellation is visited.

* **Visits per tile outliers:** Number of tiles visited that lie outside of the chosen tessellation.
		
* **Visits per time and tile** (:code:`VISITS_PER_TIME_TILE`): Compute number of times each tile of the chosen tessellation is visited per time segment. The time segment can be chosen by the user (by setting the parameter ``timewindows`` when initiating the ``DpMobilityReport``)

OD Analysis
***********

* **Origin-destination flows** (:code:`OD_FLOWS`): Compute the counts of all origin-destination combinations.
	
* **Travel time** (:code:`TRAVEL_TIME`): Histogram of travel times, bins of histogram are chosen according to input if provided.

* **Travel time quartiles:** Five number summary of travel times

* **Jump length** (:code:`JUMP_LENGTH`): The geographical straight-line distance between the trip's origin and destination.

* **Jump length quartiles:** Five number summary of jump lengths.
	

User Analysis
*************

* **Trips per user** (:code:`TRIPS_PER_USER`): Number of trips per user.
	
* **Trips per user quartiles:** Five number summary of trips per user.

* **Time between consecutive trips of a user** (:code:`USER_TIME_DELTA`): Time that passes between two consecutive trips per user.

* **Time between consecutive trips of a user quartiles:** Five number summary of user time delta.

* **Radius of gyration** (:code:`RADIUS_OF_GYRATION`): Compute the radii of gyration for all users. The radius of gyration is the characteristic distance travelled by a user, computing the spread of all locations visited by an individual around their center of mass.

* **Radius of gyration quartiles:** Five number summary of radius of gyration.

* **User tile count quartiles** (:code:`USER_TILE_COUNT`): Five number summary of user tile counts.
	
* **Mobility entropy** (:code:`MOBILITY_ENTROPY`): Compute the mobility entropy for each user. The mobility entropy is defined as the Shannon entropy of the user's visits which quantifies the probability of predicting a user's whereabouts.
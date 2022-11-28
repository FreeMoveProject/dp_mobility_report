============================================================
Analyses in the DpMobilityReport
============================================================

The *five-number summary* is a set of descriptive statistics that provides information about a dataset. It consists of the five most important sample percentiles: 
the sample minimum, the lower quartile, the median, the upper quartile and  the sample maximum.


* DS Statistics: Number of records, number of trips, number of complete trips, number of incomplete trips, number of users, number of locations

* Missing values: Number of missing values for user ID, trip ID, datetime, latitude, longitude
	
* Trips over time: Based on the time granularity of the data, the relative number of trips is computed for day, week or month segments.

* Trips per weekday: Compute the relative number of trips per weekday.

* Trips per hour: Compute the relative number of trips per hour of the day.

* Visits per tile: Compute number of times each tile of the chosen tessellation is visited.

* Visits per tile outliers: Number of tiles visited that lie outside of the chosen tessellation.
		
* Visits per tile timewindow: Compute number of times each tile of the chosen tessellation is visited per time segment. The time segment can be chosen by the user (by setting the parameter ``timewindows`` when initiating the ``DpMobilityReport``)

* OD flows: Compute the counts of all origin-destination combinations.
	
* Travel time: Histogram of travel times, bins of histogram are chosen according to input if provided.

* Travel time quartiles: Five number summary of travel times

* Jump length: The geographical straight-line distance between the trip's origin and destination.

* Jump length quartiles: Five number summary of jump lengths.
	
* Trips per user: Number of trips per user.
	
* Trips per user quartiles: Five number summary of trips per user.

* User time delta quartiles: Five number summary of user time delta. Time that passes between two consecutive trips per user.

* Radius of gyration: Compute the radii of gyration for all users. The radius of gyration is the characteristic distance travelled by a user, computing the spread of all locations visited by an individual around their centre of mass.

* Radius of gyration quartiles: Five number summary of radius of gyration.

* User tile count quartiles: Five number summary of user tile counts.
	
* Mobility entropy: Compute the mobility entropy for each user. The mobility entropy is defined as the Shannon entropy of the user's visits which quantifies the probability of predicting a user's whereabouts.
============================================================
Similarity Measures
============================================================

The similarity of two mobility reports is evaluated with a set of similarity measures. Specifically, a measure is computed for each analysis of the report.
The following table shows the analysis segments 
that can be included/exluded in the benchmark report and their corresponding analyses and similarity measures. The default measures are in bold. 
In the following the similarity measures symmetric mean average percentage error (SMAPE), Kullback-Leibler divergence (KLD), Jensen-Shannon divergence (JSD), earth mover's distance (EMD) and 
will be explained, as well as the reasoning why the specific measures are available for each analyses and how the default measure was chosen. 
The default measure is the measure most suitable for each analysis and the chosen set of default measures will be returned as the property ``similarity_measures`` when creating a benchmark report. 

Overview and default measures
********************************


.. list-table:: Analyses and their similarity measures
   :widths: 20 35 35
   :header-rows: 1

   * - Segment
     - Analysis
     - Similarity measures (Default bold)
   * - Overview
     - Dataset Statistics
     - **SMAPE**
   * - 
     - Missing values
     - **SMAPE**
   * - 
     - Trips over time
     -  SMAPE, KLD, **JSD**
   * - 
     - Trips per weekday
     - SMAPE, KLD, **JSD**
   * - 
     - Trips per hour
     - SMAPE, KLD, **JSD**
   * - Place
     - Visits per tile
     -  SMAPE, KLD, JSD, **EMD**
   * - 
     - Visits per tile quartiles
     - **SMAPE**
   * - 
     - Visits per tile outliers
     - **SMAPE**     
   * - 
     - Visits per tile ranking
     - **KT**, TOP_N
   * - 
     - Visits per time tile
     - SMAPE, KLD, JSD, **EMD**
   * - OD
     - OD flows
     - SMAPE, KLD, **JSD**   
   * -
     - OD flows ranking
     - **KT**, TOP_N     
   * -
     - OD flows quartiles
     - **SMAPE**
   * - 
     - Travel time
     - SMAPE, KLD, **JSD**, EMD
   * - 
     - Travel time quartiles
     - **SMAPE**
   * - 
     - Jump length
     - SMAPE, KLD, **JSD**, EMD
   * - 
     - Jump length quartiles
     - **SMAPE**
   * - User 
     - Trips per user
     - **EMD** TODO: KLD; JSD, SMAPE
   * -  
     - Trips per user quartiles
     - **SMAPE**
   * -  
     - User time delta
     - SMAPE, KLD, **JSD**, EMD
   * -  
     - User time delta quartiles
     - **SMAPE**
   * - 
     - Radius of gyration
     - SMAPE, KLD, **JSD**, EMD
   * - 
     - Radius of gyration quartiles
     - **SMAPE**
   * - 
     - User tile count
     - SMAPE, KLD, **JSD**, EMD
   * - 
     - User tile count quartiles
     - **SMAPE**
   * -  
     - Mobility entropy
     - SMAPE, KLD, **JSD**, EMD
   * -  
     - Mobility entropy quartiles
     - **SMAPE**
   
   



Symmetric mean absolute percentage error (SMAPE)
***************************************************


The symmetric mean absolute percentage error (SMAPE) is an accuracy measure based on percentage (or relative) errors. 
In contrast to the mean absolute percentage error, SMAPE has both a lower bound (0, meaning identical) and an upper bound (2, meaning entirely different). 

:math:`SMAPE:= \frac{1}{n} \sum_{i=1}^{n} \frac {|alternative_{i} - base_{i}|}{(|base_{i}| + |alternative_{i}|) \div 2}`, for :math:`|base_{i}| + |alternative_{i}| > 0`

SMAPE is computed for all analyses. For single counts (e.g., dataset statistics, missing values),n=1 with :math:`base_{i}` (:math:`alternative_{i}` respectively) refering to the respective count value. For the five number summary, n=5 with :math:`base_{i}` (:math:`alternative_{i}` respectively) refering to :math:`i_{th}` value of the summary. For all other analyses, n equals the number of histogram bins. 
SMAPE is employed as the default measure for single counts and for the evaluation of the five number summary, as KLD, JSD and EMD are not suitable. 


Kullback-Leibler Divergence (KLD)
**********************************
The Kullback-Leibler divergence (KLD) [1], also called relative entropy, is a widely used statistic to measure how far a probability distribution :math:`P` deviates from a reference probability distribution :math:`Q` on the same probability space :math:`\mathcal{X}`.
For discrete distributions :math:`P` and :math:`Q`, it is formally defined as 
:math:`D_{KL}(P||Q):= \sum\limits_{x \in \mathcal{X}: P(x)>0} P(x)\cdot \log \frac{P(x)}{Q(x)}`.

For example, considering a tessellation (:math:`\mathcal{X}`) the spatial distribution can be evaluated by comparing the relative counts in the synthetic data (P) per tile (:math:`x`) to the relative counts per tile in the reference dataset (Q). 
The larger the deviation of P from Q, the larger the value of the resulting KLD, with a minimum value of 0 for identical distributions.

Note that KLD is not symmetric, i.e., :math:`D_{KL}(P||Q)~\neq~D_{KL}(Q||P)`, which is why KLD is best applicable in settings with a reference model Q and a fitted model P. 
However, the lack of symmetry implies that it is not a distance metric in the  mathematical sense. 

It is worth noting that KLD is only defined if :math:`Q(x)\neq 0` for all x in the support of P, while this constraint is not required for JSD.
In practice, both KLD and JSD are computed for discrete approximations of continuous distributions, e.g., histograms approximating the relative number of trips over time based on daily or hourly counts. However, the choice of histogram bins has an impact in two respects:
Say we want to compare the number of visits per tile. Depending on the granularity of the chosen tessellation, there might be tiles with :math:`0` visits in the real dataset but :math:`>0` visits in the synthetic dataset, thus KLD would not be defined for such cases.
Additionally, the resulting values for both KLD and JSD vary according to the choice of bins, e.g., by reducing the granularity of the tessellation, the values of KLD and JSD will tend to be smaller. 

The KLD is computed for the following analyses: trips over time, trips per weekday, trips per hour, visits per tile, visits per tile timewindow, OD flows, travel time, jump length and radius of gyration.
The KLD might be ``None`` due to the fact, that it is not defined for :math:`Q(x)\neq 0` for all x in the support of P and therefore the JSD is set as a default measure.



Jensen-Shannon Divergence (JSD)
**********************************

The Jensen-Shannon divergence (JSD) solves this asymmetry by building on the KLD to calculate a symmetrical score in the sense that the divergence of P from Q is the same as Q from P: :math:`D_{JS}(P||Q) = D_{JS}(Q||P)`.
Additionally, JSD provides a smoothed and normalized version of KLD, with scores between :math:`0` (identical) and :math:`1` (maximally different) when using the base-2 logarithm, thus making it easier to relate the resulting score within a fixed finite range. 
Formally, the JSD is defined for two probability distributions :math:`P` and :math:`Q` as: :math:`D_{JS}(P||Q) := \frac{1}{2} D_{KL}(P\left\Vert\frac{P+Q}{2}\right) + \frac{1}{2} D_{KL}(Q\left\Vert\frac{P+Q}{2}\right)`.

Following this advantage of the JSD compared to the KLD, the JSD is chosen as a default measure over the KLD. 

Earth mover's distance (EMD)
********************************
Both KLD and JSD do not account for a distance of instances in the probability space :math:`\mathcal{X}`. However, the earth mover's distance (EMD) [2] between two empirical distributions allows to take a notion of distance, like the underlying geometry of the space into account. 
Informally, the EMD is proportional to the minimum amount of work required to convert one distribution into the other. 

Suppose we have a tessellation, each tile denoted by :math:`\{x_1, \ldots , x_n\}` with a corresponding notion of distance :math:`dis(x_i, x_j)` 
being the Haversine distance between the centroids of tiles :math:`x_i` and :math:`x_j`. 
For two empirical distributions :math:`P` and :math:`Q` represented by the visits in the given tiles :math:`\{p_1, \ldots , p_n\}` and :math:`\{q_1, \ldots , q_n\}`, respectively, 
the EMD can be defined as where :math:`f_{ij}` is the optimal flow that minimizes the work to transform P into Q. 

The amount of work is determined by the defined distance between instances (i.e., tiles), thus, it allows for an intuitive interpretation.
In the given example, an EMD of 100 signifies 
that on average each record of the first distribution needs to be moved 100 meters to reproduce the second distribution. On the downside, there is no fixed range as for the
JSD which provides values between 0 and 1. Thus the EMD always needs to be interpreted in the context of the dataset and the EMD of different datasets cannot be compared directly.

 
In the same manner, the EMD can be computed for histograms, by defining a distance between histogram bins. 
To measure the distance between histogram bins, the difference between the midrange values of each bin pair is computed. 
For tiles, the centroid of each tile is used to compute the haversine distance.

Thus the EMD is available for the following analyses provided in the following units: 

* visits per tile: distance in meters

* visits per time tile: average distance in meters for each timewindow

* travel time: distance in minutes

* jump length: distance in kilometers

* trips per user: distance in counts of trips

* user time delta: distance in hours

* radius of gyration: distance in kilometers

* user tile count: distance in counts of tiles

* mobility entropy: distance in mobility entropy
 

The EDM can only be computed, if a notion of distance between histogram bins or tiles can be computed. 
For example, there is no trivial distance between weekdays (you could argue that the categorization of weekdays and weekend is more important than the number of days lying inbetween). Thus, we decided to omit the EMD if there is no intuitive distance measure. 
The EMD is the default measure for visits per tile and visits per tile timewindow, as the underlying geometry is especially important to account for here. The EMD is also the default measure for the trips per user.

Kendall correlation coefficient (KT)
**************************************

The Kendall's :math:`\tau` coefficient, also known as the Kendall rank correlation coefficient, is a measure of the strength and direction of association that exists between 
two variables measured on an ordinal scale. It is a non-parametric measure of statistical associations based on the ranks of the data, i.e., the similarity of two rankings 
such as a ranking of most visited locations of two datasets. 
It returns a value between :math:`-1` and :math:`1`, where :math:`-1` means negative correlation, :math:`0` means no relationship and :math:`1` means positive correlation, 
determining the strength of association based on the pattern of concordance (ordered in the same way) and discordance (ordered differently) between all pairs, defined as follows [3]:
:math:`\tau= \frac{\textrm{number of concordant pairs} - \textrm{number of discordant pairs}}{\textrm{number of pairs}}`

Let's consider a list of locations :math:`\langle l_1,...,l_n \rangle` and let :math:`pop(D, l_i)` denote the popularity of :math:`l_i`, i.e., the number of times :math:`l_i` is visited by trajectories in dataset :math:`D` and compute the popularity :math:`pop(D_{base}, l_i)` for a base dataset and :math:`pop(D_{alt}, l_i)` for an alternative dataset for all :math:`l_i`. Then, we say that a pair of locations :math:`(l_i, l_j)` are concordant if either of the following hold:

:math:`(pop(D_{ref}, l_i) > pop(D_{ref}, l_j)) \wedge (pop(D_{syn}, l_i) > pop(D_{syn}, l_j))` or 

:math:`(pop(D_{ref}, l_i) < pop(D_{ref}, l_j)) \wedge (pop(D_{syn}, l_i) < pop(D_{syn}, l_j))`, i.e., their popularity ranks (in sorted order) agree. They are said to be discordant if their ranks disagree.

Coverage of the top n locations (TOP_N)
********************************************

The coverage of the top :math:`n` locations [4] is defined by the true positive ratio: :math:`\frac{|top_n(D_{base})\ \cap\ top_n(D_{alt})|}{n}`, where :math:`n` is the number of top locations and :math:`top_n(D_{base})` is the :math:`n` top locations of the base dataset and :math:`top_n(D_{alt})` the :math:`n` top locations of the alternative dataset.
This measure represents how well the alternative dataset is similar to the base dataset considering the most visited locations.


References:

[1] S. Kullback and R. A. Leibler. 1951. On Information and Sufficiency. The Annals of Mathematical Statistics, 22, 1, (March 1951), 79–86. doi: 10.1214/aoms/1177729694.
[2] E. Levina and P. Bickel. 2001. The Earth Mover's distance is the Mallows distance: some insights from statistics. In Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001. Volume 2. (July 2001), 251–256 vol.2. doi: 10.1109/ICCV.2001.937632.
[3] Gursoy, M. E., Liu, L., Truex, S., Yu, L., & Wei, W. (2018, October). Utility-aware synthesis of differentially private and attack-resilient location traces. In Proceedings of the 2018 ACM SIGSAC conference on computer and communications security (pp. 196-211).
[4] Bindschaedler, V., & Shokri, R. (2016, May). Synthesizing plausible privacy-preserving location traces. In 2016 IEEE Symposium on Security and Privacy (SP) (pp. 546-563). IEEE.
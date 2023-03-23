============================================================
Differential Privacy
============================================================

In the following all definitions and notions concerning differential privacy guarantees provided in this package are explained.

This mobility report creates aggregations of data. Though, despite that, the computed `statistics`_ are still vulnerable to reconstruction attacks [1].
By observing answers from measures an attacker can recover information about indiviual users.

Differential privacy provides mathematical guarantees for the privacy of an individual [2].
The concept of differential privacy is that the output of an algorithm :math:`\mathcal{A}` remains
nearly unchanged if the records of one individual are removed or added.
In this way, differential privacy limits the impact of a single individual on
the analysis outcome, preventing the reconstruction of an individual's data.


**Definition: Differential Privacy**

Let :math:`\text{Range}(\mathcal{A})` be a randomized algorithm that 
takes a mobility dataset :math:`T` as input and outputs a value from some output space :math:`\text{Range}(\mathcal{A})`.
For an :math:`\varepsilon > 0`, :math:`\mathcal{A}` is said to be 
:math:`\varepsilon`-differentially private, if for all pairs of datasets :math:`T_1` and :math:`T_2` differing in all
records of an arbitrary but fixed user, and all outputs :math:`O\subseteq \text{Range}(\mathcal{A})`,

:math:`P[\mathcal{A}(T_1) \in O] \,\leq\, \mathrm{e}^{\varepsilon} \cdot P[\mathcal{A}(T_2) \in O].`

The parameter :math:`\varepsilon` captures the privacy loss and determines
how similar the randomized outputs are based on :math:`T_1` and :math:`T_2`, and thus
specifies the impact of a single individual's data records on the output.

In differential privacy literature, the definition usually considers datasets differing in 
one record, assuming that every user makes exactly one contribution. This is for instance the case 
when 
each record of a dataset corresponds to a complete trajectory of a single user. 
However, typical mobility datasets in practice do not satisfy this assumption, containing multiple 
records 
per user. In this case, the classical definition of differential 
privacy would protect an item, such as a single trip. However, our used
definition of differential privacy protects the 
privacy of a user. We will thus distinguish between *item-level 
privacy* and *user-level privacy* [2] which can be also be modified in the :code:`DpMobilityReport` with the parameter :code:`user_privacy` (bool).


An important notion in this context is the sensitivity of a
function :math:`f` which corresponds to the maximum difference that an output can
change by removing or adding a record (item-level) or all records of a user (user-level) [2].

**Definition: Sensitivity**

Let :math:`T_1` and :math:`T_2` be two datasets differing in one record (item-level) or all records of a user 
(user-level), respectively.
The :math:`L_1`-sensitivity of :math:`f` is defined as
:math:`\Delta f = \max_{(T_1,T_2)} \vert f(T_1) - f(T_2)\vert_{1}`
for any such :math:`T_1` and :math:`T_2`.

In the mobility report, most of the functions are counts
and as such output numeric values.
For example, the function *count* outputs a single number (e.g., *trip_count*), while
*counts* outputs a number for each category and bin, respectively (e.g., *trips_per_weekday*).
A common mechanism for numeric functions is the Laplace
mechanism, where calibrated noise is added to the function's output,
drawn from a Laplace distribution :math:`Lap()` [2].

**Definition: Laplace mechanism**
Let :math:`f:T^n\to \mathbb{R}^k` with arbitrary domain :math:`T`.
The Laplace mechanism is defined as 
:math:`\mathcal{A}(T) = f(T)+ (Y_1,\ldots,Y_k)`
where :math:`Y_i` is a random variable drawn from :math:`Lap\left(\Delta f / 
\varepsilon\right)` with mean :math:`0` and variance :math:`2(\Delta
f / \varepsilon)^2`.

Note that for *counts*, :math:`k` equals the
number of categories/bins, e.g., locations, while :math:`k=1`
for the *count* function.
The magnitude of noise is calibrated according to the sensitivity :math:`\Delta f` of
a function.

If we assume item-level privacy for our report, this means for the number of
*trips over time* that noise with :math:`\Delta f=1` is added to each count,
since removing or adding one trip changes the count by one.
While this effectively hides the presence of one trip of an individual,
it does not protect the presence/absence of an individual with multiple trips.
In other words, removing or adding all trips of a user changes the number
of *trips over time* by the amount of trips of the corresponding user
instead of just one.

Suppose we are interested in the number of *OD flows*.
A user has made 10 trips. If we remove the user, the counts will always change by 10: 
If the user made all the trips between the same origin and destination, the number for this
OD flow changes by 10. The other counts remain unchanged.
If the user made the 10 trips between varying OD pairs, the counts for each respective of pair
change by 1. This means that the maximum influence of a user on such a 
measure :math:`f` and therefore its sensitivity :math:`\Delta f=10`.
In practice, however, a user can contribute an arbitrary number of trips and
the sensitivity of :math:`f` is thus unbounded.

**Bounded user contribution**

In order to limit the sensitivity,
we need to limit the number of possible trips :math:`M` of a user.
If we choose the highest number of trips a user has in our dataset for :math:`M`, we
assume local sensitivity.
Local sensitivity depends on the dataset and not only on the function.
Therefore, the local sensitivity or the maximum number of trips may jeopardize
the privacy of a user.

Sampling bounds the number of trips of a single user to :math:`M` and removes the remaining records.
The places people visit is highly predictable for most individuals [3, 4].
There are only few places that a person visits regularly,
usually the home and work place, which make up the majority of locations in a
person's mobility pattern [5].
Therefore, we assume that the global geospatial patterns of a mobility dataset
remain intact even though only a small sample of each person's trips are
included.

The maximum number of trips can be set with the parameter :code:`max_trips_per_user` in the :code:`DpMobilityReport`.
The dataset will be sampled according and the sensitivity is set respectively, if :code:`user_privacy=True`.


**Counts**
The count functions based on users have a
sensitivity of :math:`1`, since removing a user, no matter how many trips they make,
only changes one count by :math:`1`. E.g., the *trips per user* are
represented with a histogram, each bin representing a possible number of trips.
Removing/adding a user with their trips will only change the count for one bin
by :math:`1`.
The count functions based on trips have a sensitivity of
:math:`M` as explained in the previous example.
Since a trip consists of two points (start and end),
the count functions based on points have a sensitivity
of :math:`2\cdot M`.

To this end, we guarantee differentially private counts. However, e.g.,
visited locations or reported categories can reveal the identity of an 
individual.
For example, if only tiles that were actually visited are included in the
report, we can infer that all reported tiles were visited at least
once, while all tiles that were not included are not part of the dataset.
Thus, we consider all tiles within the tessellation as given to obtain a finite number of geometric shapes and 
apply the Laplace mechanism to each of the tiles, including 
those with a count of zero. Therefore, there is no certainty which tiles have actually been visited and which 
have 
not. All points outside the provided tessellation are
summarized as a single noisy count of outliers.

The tessellation defines categories for the spatial dimension.
Categories for temporal dimensions, which is reflected by 
*trips over time*, are aggregated to specific time intervals, such as
day, week or month.
Empty intervals in between should be filled with 0 values so that noise can be
applied as well. 
While the tessellation also provides a fixed bounding of the spatial extent,
there is another issue for *trips over time*:
revealing the exact first and last day of the entire time interval violates the
privacy. The same issue of leaking minimum and maximum values applies to any
count-based measure that has no 
pre-defined categories or bins, namely, 
*travel time*, *jump length*, *trips per user*, *time between trips*, *radius of 
gyration*, *locations per user* and *mobility entropy*. 
Similar to the given tessellation we can cut off bins at a defined 
minimum and maximum based on domain knowledge.
Data points outside this interval are summarized as a single noisy outliers
count.
Instead, we can also determine the minimum and maximum differentially private
to obtain bounds.
Since the minimum and maximum are included in the five-number summary we do not
need to compute them twice.

**Five-Number Summary**
The five-number summary can be returned differentially private with
the exponential mechanism [6].
This mechanism does not add noise to
the output. Instead it returns the best answer from a set based on a scoring function.
Differential privacy is 
guaranteed since sometimes an output is returned even though it has not the highest 
score. 

**Definition: Exponential mechanism**
Given an input dataset :math:`T` the Exponential mechanism [6]
randomly samples an output :math:`O\subseteq \text{Range}(\mathcal{A})` with a
probability	proportional to :math:`e^{\frac{\varepsilon s(T,O)}{2\Delta s}}`,
where :math:`s` is the scoring function and :math:`\Delta s` the corresponding
sensitivity.

We use the exponential mechanism to determine the five-number summary including
the minimum and maximum. In this case the scoring function is a rank function of
the sorted input.
Since we determine the index of an element, a user with :math:`M` trips influences
the output by :math:`M`. Therefore, the sensitivity for the five-number summary is the same as that for 
counts.
Note that the element returned by the exponential mechanism is always a member
of the set :math:`\text{Range}(\mathcal{A})`.
This is reasonable for a finite set where a noisy response is not useful.


References:

[1] Dwork, C., and A. Roth. 2013. “The Algorithmic Foundations of Differential Privacy.” Foundations and Trends in Theoretical Computer Science 9 (3–4): 211–407. doi:10.1561/ 0400000042.

[2] Dwork, C., F. McSherry, K. Nissim, and A. Smith. 2006. “Calibrating Noise to Sensitivity in Private Data Analysis.” TCC '06: Proceedings of the 3rd Theory of Cryptography Conference, New York, United States, 265–284. Springer.

[3] Gonzalez, M. C., C. Hidalgo, and A.-L. Barabasi. 2008. “Understanding Individual Human Mobility Patterns.” Nature 453 (7196): 779–782. doi:10.1038/nature06958.

[4] Song, C., Z. Qu, N. Blumm, and A.-L. Barabási. 2010. “Limits of Predictability in Human Mobility.” Science 327 (5968): 1018–1021. doi:10.1126/science.1177170.

[5] Do, T. M. T., and D. Gatica-Perez. 2014. “The Places of Our Lives: Visiting Patterns and Automatic Labeling from Longitudinal Smartphone Data.” IEEE Transactions on Mobile Computing 13 (3): 638–648. doi:10.1109/TMC.2013.19.

[6] McSherry, F. and K. Talwar. 2007. “Mechanism Design via Differential Privacy.” FOCS ’07: Proceedings of the 48th Annual IEEE Symposium on Foundations of Computer Science, Providence, RI, 94–103. IEEE Computer Society.


.. _`statistics`: https://dp-mobility-report.readthedocs.io/en/latest/analyses.html
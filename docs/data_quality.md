# Data quality: findings and decisions

Source: the full run on 18 Nov 2026 over all train (80) and student_test (10) trips.
Numbers come from `outputs/eda/trip_stats.csv`, `outputs/clean/cleaning_summary.csv` and
`outputs/clean/cleaning_config.json`. The evaluator partition was not read.

The data is the Kubicka et al. (2015) public map-matching dataset. Its reference routes were
corrected by hand and are known to contain some errors, so we treat them as reference labels,
not as verified ground truth.

## Dataset at a glance

| | train | student_test |
|---|---|---|
| Trips | 80 | 10 |
| GPS points | 208,417 | 19,160 |
| Points per trip (median, min to max) | 1,529 (361 to 19,313) | 1,420 (262 to 3,958) |
| Road arcs per trip (median) | 19,527 | 13,145 |

- Trips are spread over 25 UTM zones (longitude about −100° to 137°, latitude about −35° to 62°).
  **Decision:** project each trip to its own UTM zone; never measure distance in degrees.
- Sampling is nominally 1 Hz: the median trip has 99.1% of steps exactly 1 s apart. But 5 trips
  (00000022, 00000004, 00000090, 00000039, 00000046) have only 53–62% 1-second steps.
  **Decision:** use `dt` explicitly in speed and transition calculations rather than assuming 1 s.

## GPS track issues

| Issue | Size | Decision |
|---|---|---|
| NaN, out-of-range coordinates, duplicate or backwards timestamps | 0 in all 90 trips | No action needed; checks kept in the pipeline |
| Gaps over 30 s | 135 gaps in 30 trips. Extreme cases: 00000090 (about 42 h), 00000078 (about 14.7 h), 00000069 (44 min), 00000021 (36 min) | Split trips at gaps over 180 s (Newson & Krumm): 108 segments in train, 11 in student_test. The HMM decodes each segment separately |
| Speeds above 50 m/s (180 km/h) | 1,538 steps in 42 trips, mostly sustained (e.g. 00000032: 892 steps, 95th percentile 237 km/h) | Kept. Sustained high speed looks like genuine motorway driving, so a plain speed threshold would discard real data. Only out-and-back spikes count as errors: 5 points in train, 0 in student_test |
| Repeated identical positions | 578 points in 22 trips | Flagged only, as a likely stationary vehicle |
| Points within 2σ of the previous kept point | About 51% of train points | Flagged only. At 1 Hz, Newson & Krumm's filter would remove half the data, and the task needs a prediction for every point |

## Road network issues

| Issue | Size | Decision |
|---|---|---|
| Network split into several disconnected pieces | 89 of 90 trips; median 19 pieces, but the largest piece holds a median 98% of nodes (minimum 69%) | Transitions between unconnected candidates get a small fallback probability instead of zero, so Viterbi can't dead-end |
| One-way roads | Median 95% of arcs have a reverse arc; the lowest trip has 35% | Heading-vs-arc-direction is a feature; arc direction matters |
| Duplicate arcs | 1,903 in 60 trips | De-duplicate when building candidates so the same road isn't scored twice |
| Self-loops / zero-length arcs | 24 in 9 trips / 106 in 32 trips | Excluded from candidates (no direction, no length) |
| Arcs pointing to missing nodes, isolated nodes | 0 | No action |

## Reference route issues

| Issue | Size | Decision |
|---|---|---|
| Reference route doesn't follow the GPS | 00000054 (88% of points more than 30 m off, median 139 m) and 00000033 (71%, median 122 m) | Flagged as suspect reference trips. Left out of the noise estimate and out of training labels |
| Smaller mismatches | 00000002 (14%) and 00000043 (13%) of points more than 30 m off | Points beyond 40 m from the route are left out of training labels (`label_exclude_far`: 23,625 train points, 38 student_test) |
| Consecutive route arcs that don't connect | 14 breaks in 11 routes (≥ 98% of steps still connected) | Accepted as hand-correction errors. When we evaluate route continuity, we report these trips separately |

## Values estimated from training data

- **GPS noise σz = 6.68 m:** for each train trip, 1.4826 × the median GPS-to-route distance, then
  the median across the 80 trips. Per-trip values range from 5.1 m to 205 m; the extremes are the
  two suspect trips. Taking the median across trips keeps them from distorting the estimate.
- **Label cut-off = 40 m** (6σz).
- **Candidate settings** (train, clean points within the cut-off): the reference arc is the
  single nearest arc only 90.3% of the time. k = 5 nearest arcs misses 0.96%, k = 10 misses 0.33%.
  A 30 m radius misses 0.95%. These numbers feed Person B's choice of k and radius.

Every threshold above was fitted on train only and applied unchanged to student_test.

# Data-quality findings (auto-generated counts)

Partitions read: student_test, train

- Trips with NaN values: none
- Trips with out-of-range coordinates: none
- Trips with non-increasing timestamps: none
- Trips with a GPS gap over 30 s: 30 trips (train/00000002, train/00000004, train/00000006, train/00000011, train/00000014, train/00000021, train/00000024, train/00000028, train/00000032, train/00000037, train/00000041, train/00000042, train/00000043, train/00000044, train/00000054 ...)
- Trips with jumps over 180 km/h: 42 trips (train/00000001, train/00000005, train/00000008, train/00000011, train/00000012, train/00000013, train/00000019, train/00000020, train/00000024, train/00000025, train/00000028, train/00000031, train/00000032, train/00000033, train/00000038 ...)
- Trips whose arcs reference missing nodes: none
- Trips whose network has >1 weak component: 89 trips (train/00000001, train/00000002, train/00000003, train/00000004, train/00000005, train/00000006, train/00000008, train/00000010, train/00000011, train/00000012, train/00000013, train/00000014, train/00000016, train/00000018, train/00000019 ...)
- Trips with isolated nodes: none
- Routes with disconnected consecutive arcs: 11 trips (train/00000002, train/00000019, train/00000033, train/00000041, train/00000042, train/00000069, train/00000075, train/00000080, train/00000086, train/00000089, student_test/00000076)
- Routes with out-of-range arc ids: none
- Trips with >10% of GPS fixes over 30 m from the reference route: 4 trips (train/00000002, train/00000033, train/00000043, train/00000054)

Overall (median across trips):

- n_points: 1496.50
- duration_s: 1710.00
- pct_dt_exactly_1s: 99.14
- max_gap_s: 7.00
- path_km: 24.05
- median_speed_kmh: 46.09
- n_nodes: 9305.00
- n_arcs: 18761.50
- pct_arcs_bidirectional: 95.11
- gps_to_net_median_m: 4.16
- gps_to_route_median_m: 4.52

Add the team's own interpretation and decisions below.

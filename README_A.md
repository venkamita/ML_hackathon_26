# Person A: data loading, EDA and Folium maps

## Setup (once)

```
cd "D:\Dowloads\Downloads\Student folder\pathsnap"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Needs Python 3.10 or newer.

## Run

```
python -m eda.run_eda --data-root ..
```

`--data-root` is the folder that holds `train/` and `student_test/` (or its parent).
To map specific trips: `--map-trips train/00000005 student_test/00000052`.

Then the cleaning step:

```
python -m data_processing.clean --data-root ..
```

## Cleaning step (outputs/clean/)

Adapted from Newson & Krumm (2009), the Chao et al. (2019) survey, Map Matching 2,
and the notes on the Kubicka et al. (2015) dataset (which this data comes from).
No point is deleted, because a prediction is needed for every GPS point. Each point gets flags instead.

| Column | Meaning | Used for |
|---|---|---|
| flag_nan, flag_out_of_range, flag_bad_time | Broken values or timestamps | Hard error |
| flag_spike | Out-and-back jump faster than 50 m/s | Hard error |
| is_clean | None of the hard errors | Training filter |
| flag_duplicate_pos | Same lon/lat as an earlier point | Info |
| flag_speed_over_50mps | Faster than 50 m/s (genuine motorway driving does this) | Info |
| flag_redundant_2sigma | Within 2*sigma_z of the previous kept point (Newson & Krumm filter) | Optional down-sampling |
| segment_id | Increments after any gap > 180 s | HMM: decode each segment separately |
| label_dist_to_route_m, label_route_rank, label_exclude_far | Distance to the reference route, number of arcs closer than it, beyond cut-off | **Label quality only. Never use as model features.** |

- `cleaning_config.json`: sigma_z, the thresholds, and trips whose reference route looks wrong. All fitted on train only.
- `sigma_per_trip_train.csv`: the noise estimate for each training trip.
- `candidate_miss_rate_by_k_train.csv`, `candidate_miss_rate_by_radius_train.csv`, `plots/candidate_miss_rate_train.png`:
  how often the reference arc would be missed for each k / radius. Give these to Person B for choosing candidate settings.
  They are computed on clean points inside the label cut-off.
- `tracks/<partition>_<trip>.csv`: every point with its flags (about 50 MB in total).

## Outputs (outputs/eda/)

| File | What it is |
|---|---|
| inventory.csv | Files per trip, sizes, row counts |
| trip_stats.csv | One row per trip: GPS, network, route and GPS-to-road stats |
| summary_by_partition.csv | describe() of every stat, split by partition |
| missing_invalid_values.csv | NaN / out-of-range / duplicate / jump counts |
| data_quality_notes.md | Auto-flagged issues; add your own interpretation |
| plots/*.png | 8 EDA plots for the report |
| maps/*.html | Folium maps with layer control (open in a browser) |

When reading the CSVs, use `dtype={"trip_id": str}` to keep the leading zeros.

## Shared code for the team

- `common/data_loader.py`: `load_trip(root, partition, trip_id)`, `list_trips`, `file_inventory`
- `common/geo.py`: UTM projection (`add_xy`), `arc_geometry`, `track_kinematics`,
  `point_to_segments` (distance and projection fraction), `angle_diff_deg`
- `viz/folium_map.py`: `make_trip_map(trip, out_html, predicted_arcs=None)`.
  Pass the predicted arc ids later to add the predicted-route layer.

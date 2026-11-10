# LeRobot dataset format written by `rebot-record`

Target: **LeRobotDataset `codebase_version` "v3.0"**, as written and read by `lerobot==0.6.1`
(PyPI, 2026-08-03). `lerobot==0.7.0` (2026-11-05) keeps `CODEBASE_VERSION = "v3.0"`
(`lerobot/datasets/dataset_metadata.py`). v3.0 is not readable by the v2.1 loaders (lerobot < 0.4).

Sources: the `lerobot-0.6.1` wheel (`lerobot/datasets/utils.py`, `dataset_metadata.py`,
`compute_stats.py`, `video_utils.py`, `configs/video.py`, `utils/constants.py`), plus a reference
dataset written by `LeRobotDataset.create(...).add_frame(...).save_episode()` with lerobot 0.6.1
and inspected with pyarrow.

## Layout

```
<root>/
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  data/chunk-{chunk:03d}/file-{file:03d}.parquet
  videos/<video_key>/chunk-{chunk:03d}/file-{file:03d}.mp4
```

LeRobot packs several episodes into one data/video file (up to 100 MB / 200 MB) and stores each
episode's position in `meta/episodes`. `rebot-record` writes one data file and one video file per
episode (`file_index = episode_index % 1000`, `chunk_index = episode_index // 1000`, video
`from_timestamp = 0`); the loader resolves files through the episode rows, so both packings load.

## meta/info.json

`codebase_version`, `fps` (int), `features`, `total_episodes`, `total_frames`, `total_tasks`,
`chunks_size` (1000), `data_files_size_in_mb` (100), `video_files_size_in_mb` (200), `data_path`,
`video_path`, `robot_type`, `splits` (`{"train": "0:<total_episodes>"}`).

Each feature: `{"dtype", "shape", "names"}`. Video features have `dtype: "video"`,
`shape: [H, W, 3]`, `names: ["height", "width", "channels"]` and an `info` block
(`video.height/width/codec/pix_fmt/fps/channels`, `has_audio`, `video.g`, `video.crf`,
`video.preset`, `video.fast_decode`, `video.video_backend`, `video.extra_options`, `is_depth_map`).

Default features, filled by the writer (`lerobot/utils/constants.py` `DEFAULT_FEATURES`):

| key | dtype | shape |
|---|---|---|
| `timestamp` | float32 | [1] (stored as a scalar column; `frame_index / fps`) |
| `frame_index` | int64 | [1] (0-based within the episode) |
| `episode_index` | int64 | [1] |
| `index` | int64 | [1] (0-based across the dataset) |
| `task_index` | int64 | [1] (row in `meta/tasks.parquet`) |

## data parquet

One row per frame. Vector features are `fixed_size_list<float>[N]`; default features are
scalars. The schema metadata key `huggingface` carries the `datasets` feature description
(`List` with `length`, `Value`). Video frames are not in the parquet.

## meta/episodes parquet

One row per episode: `episode_index`, `tasks` (list<string>), `length`, `data/chunk_index`,
`data/file_index`, `dataset_from_index`, `dataset_to_index`, and per video key
`videos/<key>/chunk_index`, `videos/<key>/file_index`, `videos/<key>/from_timestamp`,
`videos/<key>/to_timestamp`; then `stats/<feature>/<stat>` (list<double>, `count` list<int64>)
and `meta/episodes/chunk_index`, `meta/episodes/file_index`.

## meta/tasks.parquet

Column `task_index` (int64) with the task string as the pandas index named `task` (pandas
metadata in the schema). `load_tasks` uses `pd.read_parquet`, so the index metadata is required.

## meta/stats.json

Per feature: `min`, `max`, `mean`, `std`, `count`, `q01`, `q10`, `q50`, `q90`, `q99`. Vector
features have shape [N]; video features are per channel, shape [3, 1, 1], scaled to [0, 1]
(pixel / 255). Dataset stats aggregate per-episode stats: count-weighted mean, pooled variance,
min/max, count-weighted quantiles (`aggregate_feature_stats`).

## Video

LeRobot 0.6.1 encodes with PyAV: `vcodec="libsvtav1"`, `pix_fmt="yuv420p"`, `g=2`, `crf=30`,
`preset=12`. It also accepts `h264`, `hevc`, `libaom-av1` and hardware codecs. Decoding defaults
to `torchcodec` when it imports, else PyAV. A frame is looked up at
`from_timestamp + timestamp` with `tolerance_s = 1e-4`, so the video must hold exactly one frame
per data row at `pts = frame_index / fps`.

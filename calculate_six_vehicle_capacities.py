"""Reproduce paper Eq. (1) for six vehicles, with separate quality diagnostics.

Run from the project root: py -3.11 -X utf8 calculate_six_vehicle_capacities.py
Only NumPy, pandas and matplotlib are required. Original data are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

VEHICLES = [
    "LK6ADAE22LB201439-1", "LK6ADAE22LB306711_1", "LK6ADAE28LB200991_1",
    "LK6CCAE3XLE551145_1", "LK6CCAE31LE554709_1", "LK6CCAE36LE550543_1",
]
FIELDS = ["发送时间", "充电状态", "总电流", "SOC", "累计里程", "总电压",
          "可充电储能子系统各温度探针检测到的温度值"]
CORE = ["charge_state", "current_a", "soc", "voltage_v", "mileage_km", "avg_temp"]
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class HashingReader(io.RawIOBase):
    """Hash the exact source bytes consumed by pandas, without a second read."""

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
            self.bytes_read += count
        return count


def code(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str.split(":", n=1).str[0], errors="coerce")


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Both slash dates and compact YYYYmmddHHMMSS occur in the six vehicles."""
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for pattern, fmt in [(r"\d{14}", "%Y%m%d%H%M%S"),
                         (r"\d{4}/.*", "%Y/%m/%d %H:%M:%S"),
                         (r"\d{4}-.*", "%Y-%m-%d %H:%M:%S")]:
        mask = series.str.fullmatch(pattern, na=False)
        result.loc[mask] = pd.to_datetime(series.loc[mask], format=fmt, errors="coerce")
    bad = series.ne("") & result.isna()
    if bad.any():
        raise ValueError(f"Unrecognized timestamps: {series.loc[bad].head().tolist()}")
    return result


def mean_probes(value: str) -> float:
    values = [float(x) for x in NUMBER.findall(value)]
    return float(np.mean(values)) if values else np.nan


def decode(frame: pd.DataFrame, file_number: int, offset: int) -> pd.DataFrame:
    present = frame[FIELDS[0]].ne("")
    source_rows = np.arange(offset + 2, offset + len(frame) + 2)[present]
    frame = frame.loc[present].copy()
    timestamps = parse_timestamps(frame[FIELDS[0]])
    current_code, soc, voltage, mileage = [
        code(frame[name]) for name in ["总电流", "SOC", "总电压", "累计里程"]]
    probe_values = {value: mean_probes(value) for value in frame[FIELDS[6]].unique()}
    states = {"停车充电": 1, "行驶充电": 2, "未充电状态": 3, "充电完成": 4}
    return pd.DataFrame({
        "time_s": timestamps.to_numpy(dtype="datetime64[s]").astype("int64"),
        "source_file": file_number, "source_row": source_rows,
        "charge_state": frame[FIELDS[1]].map(states).fillna(-1).to_numpy(),
        "current_a": ((current_code - 10000) / 10).where(current_code.between(0, 65533)).to_numpy(),
        "soc": soc.where(soc.between(0, 100)).to_numpy(),
        "voltage_v": (voltage / 10).where(voltage.between(0, 65533)).to_numpy(),
        "mileage_km": (mileage / 10).where(mileage.between(0, 4294967293)).to_numpy(),
        "avg_temp": frame[FIELDS[6]].map(probe_values).to_numpy(),
    })


def relevant(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep every charge point plus transitions and file/chunk endpoints."""
    if frame.empty:
        return frame
    charge = frame.charge_state.eq(1)
    keep = charge | charge.shift(1, fill_value=False) | charge.shift(-1, fill_value=False)
    keep.iloc[0] = keep.iloc[-1] = True
    return frame.loc[keep].copy()


def read_vehicle(root: Path, folder: str, destination: Path, limit: int | None) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted((root / folder).glob("*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(folder)
    old_manifest = root / "raw_vehicle_reproduction_20261110/data/raw_manifest.json"
    reuse = {}
    if folder == "LK6CCAE31LE554709_1" and old_manifest.exists():
        manifest = json.loads(old_manifest.read_text(encoding="utf-8"))
        if Path(manifest["raw_directory"]).resolve() == (root / folder).resolve():
            reuse = {Path(record["source_file"]).name: record for record in manifest["files"]}
    pieces, audit = [], []
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        number = int(path.stem.rsplit("_", 1)[1])
        before = path.stat()
        cached = reuse.get(path.name)
        if cached:
            if before.st_size != cached["source_bytes"] or before.st_mtime_ns != cached["source_mtime_ns"]:
                raise ValueError(f"Cached source metadata changed: {path}")
            cache = Path(cached["cache_file"])
            if hashlib.sha256(cache.read_bytes()).hexdigest() != cached["cache_sha256"]:
                raise ValueError(f"Existing numeric cache hash mismatch: {cache}")
            with np.load(cache) as data:
                frame = pd.DataFrame({name: data[name] for name in ["time_s", "source_file", "source_row", *CORE]})
            pieces.append(relevant(frame))
            record = dict(cached)
            record["read_mode"] = "verified_existing_numeric_cache"
        else:
            offset = missing = charging = telemetry = negative_steps = 0
            previous_time = None
            with path.open("rb") as raw:
                hashed = HashingReader(raw)
                buffered = io.BufferedReader(hashed, buffer_size=1024**2)
                for chunk in pd.read_csv(buffered, encoding="utf-8-sig", usecols=FIELDS,
                                         dtype=str, keep_default_na=False, chunksize=50000):
                    missing += int(chunk[FIELDS[0]].eq("").sum())
                    frame = decode(chunk, number, offset)
                    offset += len(chunk)
                    telemetry += len(frame)
                    charging += int(frame.charge_state.eq(1).sum())
                    if len(frame):
                        times = frame.time_s.to_numpy()
                        negative_steps += int((np.diff(times) < 0).sum())
                        negative_steps += int(previous_time is not None and times[0] < previous_time)
                        previous_time = times[-1]
                    # Some vehicles contain strongly shuffled timestamps. Keep
                    # noncharging samples until the complete timeline is sorted;
                    # dropping them earlier can silently merge distinct charges.
                    frame = frame.astype({"source_file": "uint16", "source_row": "uint32",
                                          "charge_state": "int8"})
                    pieces.append(frame)
                digest, bytes_read = hashed.digest.hexdigest(), hashed.bytes_read
            if bytes_read != before.st_size or path.stat().st_mtime_ns != before.st_mtime_ns:
                raise ValueError(f"Source changed or was not fully read: {path}")
            record = {"source_file": str(path), "file_number": number,
                      "source_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
                      "source_sha256": digest, "raw_rows": offset, "telemetry_rows": telemetry,
                      "charging_rows": charging, "missing_timestamp_rows": missing,
                      "negative_time_steps": negative_steps, "read_mode": "raw_csv_hashed_full_stream"}
        audit.append(record)
        if index == 1 or index % 10 == 0 or index == len(files):
            message = f"{folder}: {index}/{len(files)} files, {time.perf_counter()-started:.1f}s"
            (destination / "progress.txt").write_text(message, encoding="utf-8")
            print(message, flush=True)
    # Persist only the small, reproducible charging subset, never full telemetry.
    result = pd.concat(pieces, ignore_index=True)
    del pieces
    result = result.sort_values("time_s", kind="stable").reset_index(drop=True)
    result = relevant(result).reset_index(drop=True)
    np.savez_compressed(destination / "charging_points.npz", **{name: result[name].to_numpy() for name in result})
    return result, audit


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.sort_values(["time_s", "source_file", "source_row"], kind="stable").reset_index(drop=True)
    duplicate = frame.duplicated("time_s", keep=False)
    conflict_times = []
    if duplicate.any():
        conflicts = frame.loc[duplicate].groupby("time_s")[CORE].nunique(dropna=False).gt(1).any(axis=1)
        conflict_times = conflicts.index[conflicts].to_numpy()
    result = frame.drop_duplicates("time_s", keep="first").copy()
    result["conflicting_duplicate"] = result.time_s.isin(conflict_times)
    return result.reset_index(drop=True), {"duplicate_rows_removed": len(frame)-len(result),
                                          "conflicting_timestamp_groups": len(conflict_times)}


def integrate(points: pd.DataFrame, segment_id: int, max_gap: float,
              left_observed: bool, right_observed: bool) -> dict:
    t = points.time_s.to_numpy(float)
    current, soc = points.current_a.to_numpy(float), points.soc.to_numpy(float)
    dt = np.diff(t)
    if len(points) < 2 or not np.isfinite(np.c_[t, current, soc]).all() or (dt <= 0).any():
        raise ValueError("Invalid segment supplied to integration")
    interval_charge = -(current[:-1] + current[1:]) * dt / 2 / 3600
    charge = float(interval_charge.sum())
    span = float(soc[-1] - soc[0])
    temp = points.avg_temp.to_numpy(float)
    good_temp = np.isfinite(temp[:-1]) & np.isfinite(temp[1:])
    mean_temp = (float(np.sum((temp[:-1][good_temp] + temp[1:][good_temp]) / 2 * dt[good_temp]) /
                       dt[good_temp].sum()) if good_temp.any() else np.nan)
    flags = []
    if dt.max() > max_gap:
        flags.append("sampling_gap_over_threshold")
    if not left_observed:
        flags.append("left_boundary_unobserved")
    if not right_observed:
        flags.append("right_boundary_unobserved")
    if np.diff(soc).min() < -1:
        flags.append("soc_downward_jump_over_1pp")
    if np.diff(soc).max() > 5:
        flags.append("soc_upward_jump_over_5pp")
    if points.voltage_v.isna().any() or points.voltage_v.le(0).any():
        flags.append("nonpositive_or_missing_pack_voltage")
    if points.mileage_km.max() - points.mileage_km.min() > 1:
        flags.append("parked_charge_mileage_span_over_1km")
    if (current > 0).any():
        flags.append("contains_discharge_current")
    eligible = span > 25 and charge > 0
    return {"segment_id": segment_id,
            "start_time": str(pd.to_datetime(int(t[0]), unit="s")),
            "end_time": str(pd.to_datetime(int(t[-1]), unit="s")),
            "sample_count": len(points), "duration_hours": float(dt.sum()/3600),
            "start_soc": soc[0], "end_soc": soc[-1], "delta_soc": span,
            "charge_ah": charge, "capacity_ah": 100*charge/span if span > 0 else np.nan,
            "right_endpoint_capacity_ah": float(np.sum(-current[1:]*dt)/36/span) if span > 0 else np.nan,
            "avg_temp": mean_temp, "mileage_km": float(points.mileage_km.iloc[-1]),
            "max_interval_s": float(dt.max()), "gap_count": int((dt > max_gap).sum()),
            "gap_duration_seconds": float(dt[dt > max_gap].sum()),
            "negative_soc_steps": int((np.diff(soc) < 0).sum()),
            "first_source_file": int(points.source_file.iloc[0]),
            "first_source_row": int(points.source_row.iloc[0]),
            "last_source_file": int(points.source_file.iloc[-1]),
            "last_source_row": int(points.source_row.iloc[-1]),
            "paper_eligible": eligible, "quality_eligible": eligible and not flags,
            "quality_flags": ";".join(flags)}


def calculate_segments(frame: pd.DataFrame, max_gap: float, split_gaps: bool) -> pd.DataFrame:
    """Paper view retains gaps and flags them; gap-aware view never bridges gaps."""
    charge = frame.charge_state.eq(1).to_numpy()
    valid = (np.isfinite(frame[["current_a", "soc"]].to_numpy(float)).all(axis=1)
             & ~frame.conflicting_duplicate.to_numpy())
    eligible = charge & valid
    times = frame.time_s.to_numpy(float)
    dt = np.r_[np.inf, np.diff(times)]
    breaks = np.r_[True, eligible[1:] != eligible[:-1]]
    if split_gaps:
        breaks |= dt > max_gap
    boundaries = np.r_[np.flatnonzero(breaks), len(frame)]
    records = []
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        if not eligible[first] or last-first < 2:
            continue
        left = bool(first > 0 and not charge[first-1] and valid[first-1] and dt[first] <= max_gap)
        right = bool(last < len(frame) and not charge[last] and valid[last] and dt[last] <= max_gap)
        record = integrate(frame.iloc[first:last], len(records)+1, max_gap, left, right)
        records.append(record)
    return pd.DataFrame(records)


def process_vehicle(root: str, folder: str, output: str, max_gap: float, limit: int | None,
                    resume: bool) -> dict:
    root, output = Path(root), Path(output)
    destination = output / folder
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination / "audit.json"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if completed.exists():
        audit = json.loads(completed.read_text(encoding="utf-8"))
        if resume and audit["script_sha256"] == source_hash and audit["max_gap_seconds"] == max_gap and audit["limit_files"] == limit:
            for record in audit["files"]:
                stat = Path(record["source_file"]).stat()
                if stat.st_size != record["source_bytes"] or stat.st_mtime_ns != record["source_mtime_ns"]:
                    raise ValueError("Resume source metadata mismatch")
            for name, digest in audit["output_sha256"].items():
                if hashlib.sha256((destination/name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Resume output hash mismatch")
            print(f"{folder}: verified completed result reused", flush=True)
            return audit["summary"]
        raise FileExistsError(f"Use a fresh output folder: {destination}")
    points, source_audit = read_vehicle(root, folder, destination, limit)
    points, duplicates = deduplicate(points)
    paper = calculate_segments(points, max_gap, split_gaps=False)
    gaps = calculate_segments(points, max_gap, split_gaps=True)
    paper_kept = paper.loc[paper.paper_eligible].copy().reset_index(drop=True)
    clean = paper.loc[paper.quality_eligible].copy().reset_index(drop=True)
    gaps_kept = gaps.loc[gaps.paper_eligible].copy().reset_index(drop=True)
    outputs = {"all_charging_sessions.csv": paper, "paper_capacity.csv": paper_kept,
               "quality_capacity.csv": clean, "all_gap_aware_segments.csv": gaps,
               "gap_aware_capacity.csv": gaps_kept}
    for name, result in outputs.items():
        result.insert(0, "vehicle", folder)
        result.insert(1, "cycle_index", np.arange(1, len(result)+1))
        result.to_csv(destination / name, index=False, encoding="utf-8-sig", float_format="%.12g")
    values = paper_kept.capacity_ah
    summary = {"vehicle": folder, "file_count": len(source_audit),
               "telemetry_rows": sum(item["telemetry_rows"] for item in source_audit),
               "source_gb": sum(item["source_bytes"] for item in source_audit)/1024**3,
               "charging_sessions": len(paper), "paper_cycles": len(paper_kept),
               "quality_cycles": len(clean), "gap_aware_cycles": len(gaps_kept),
               "soc_exactly_25pp_excluded": int((paper.delta_soc.eq(25) & paper.charge_ah.gt(0)).sum()),
               "flagged_paper_cycles": int(paper_kept.quality_flags.ne("").sum()),
               "start_time": paper_kept.start_time.min(), "end_time": paper_kept.end_time.max(),
               "median_capacity_ah": float(values.median()), "min_capacity_ah": float(values.min()),
               "max_capacity_ah": float(values.max()),
               "quality_median_ah": float(clean.capacity_ah.median()),
               "quality_min_ah": float(clean.capacity_ah.min()),
               "quality_max_ah": float(clean.capacity_ah.max())}
    audit = {"summary": summary, **duplicates, "script_sha256": source_hash,
             "charging_cache_schema": 2,
             "charging_cache_sha256": hashlib.sha256((destination/"charging_points.npz").read_bytes()).hexdigest(),
             "paper_pdf_sha256": hashlib.sha256((root/"225112520141472058.pdf").read_bytes()).hexdigest(),
             "max_gap_seconds": max_gap, "limit_files": limit,
             "nonfinite_or_conflicting_charge_points": int((points.charge_state.eq(1) &
                (~np.isfinite(points[["current_a", "soc"]].to_numpy(float)).all(axis=1) |
                 points.conflicting_duplicate)).sum()),
             "output_sha256": {name: hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in outputs},
             "files": source_audit}
    completed.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE {folder}: paper={len(paper_kept)}, quality={len(clean)}, gap-aware={len(gaps_kept)}", flush=True)
    return summary


def make_plots(output: Path, folders: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "none"})
    colors = ["#2463A5", "#CE6A31", "#258679", "#9B5594", "#BC9131", "#5C6BB1"]
    for view, filename, label in [("paper_capacity.csv", "six_vehicles_paper_capacity", "论文公式容量（ΔSOC > 25%）"),
                                  ("quality_capacity.csv", "six_vehicles_quality_capacity", "通过数据质量检查的容量"),
                                  ("gap_aware_capacity.csv", "six_vehicles_gap_aware_capacity", "断开超过30秒缺报后的容量")]:
        combined = []
        for x_mode in ["cycle", "date"]:
            fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.4), constrained_layout=True)
            fig.suptitle(f"六辆车充电容量曲线 | {label}", fontsize=18, fontweight="bold")
            for index, (folder, ax) in enumerate(zip(folders, axes.flat)):
                frame = pd.read_csv(output / folder / view)
                if x_mode == "cycle":
                    combined.append(frame)
                    x = frame.cycle_index
                else:
                    x = pd.to_datetime(frame.end_time)
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.plot(x, frame.capacity_ah, color=colors[index], linewidth=0.75, alpha=0.85)
                ax.set_title(f"V{index+1}  {folder[:17]}  |  n={len(frame):,}", loc="left", fontsize=11)
                ax.set_xlabel("有效充电片段序号" if x_mode == "cycle" else "日期")
                ax.set_ylabel("估算容量 / Ah")
                ax.grid(alpha=0.18)
                if frame.empty:
                    ax.text(.5, .5, "无符合条件的片段", transform=ax.transAxes, ha="center")
            suffix = "" if x_mode == "cycle" else "_by_date"
            fig.savefig(output / f"{filename}{suffix}.png", dpi=190, facecolor="white")
            fig.savefig(output / f"{filename}{suffix}.svg", facecolor="white")
            plt.close(fig)
        pd.concat(combined, ignore_index=True).to_csv(output / f"all_vehicles_{view}", index=False, encoding="utf-8-sig")
    # Individual figures preserve original estimates and the separate quality view.
    for index, folder in enumerate(folders):
        raw = pd.read_csv(output / folder / "paper_capacity.csv")
        clean = pd.read_csv(output / folder / "quality_capacity.csv")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        fig.suptitle(f"V{index+1}  {folder[:17]} | 充电容量估算", fontsize=15)
        for ax, frame, title in zip(axes, [raw, clean], ["论文公式，全部有效片段", "通过数据质量检查的片段"]):
            ax.plot(pd.to_datetime(frame.end_time), frame.capacity_ah, color=colors[index], lw=.9)
            ax.set_title(f"{title}（n={len(frame)}）", loc="left", fontsize=10)
            ax.set_ylabel("容量 / Ah")
            ax.grid(alpha=.2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].set_xlabel("日期")
        fig.savefig(output / folder / "capacity_curve.png", dpi=190, facecolor="white")
        fig.savefig(output / folder / "capacity_curve.svg", facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("output/six_vehicle_capacity_20261111"))
    parser.add_argument("--vehicles", nargs="+", default=VEHICLES, choices=VEHICLES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-gap-seconds", type=float, default=30)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        summaries = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_vehicle, str(args.root.resolve()), folder,
                                   str(args.output.resolve()), args.max_gap_seconds,
                                   args.limit_files, args.resume) for folder in args.vehicles]
            for future in as_completed(futures):
                summaries.append(future.result())
                pd.DataFrame(summaries).to_csv(args.output / "vehicle_summary_partial.csv", index=False, encoding="utf-8-sig")
        summary = pd.DataFrame(summaries).set_index("vehicle").loc[args.vehicles].reset_index()
        summary.to_csv(args.output / "vehicle_summary.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    make_plots(args.output, args.vehicles)
    print(f"ALL COMPLETE: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

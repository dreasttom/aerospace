
"""
This code is provided as is. I think it is useful, but you should certainly feel free to make adjustments
as you see fit. The goal is a tool that can produce simulated radar data for whatever you may need.
That could be for machine learning, or other purposes.
- Supports:
    * configurable aircraft count
    * weather severity
    * jamming severity
    * false returns / clutter
    * parameter selection for export
    * live radar display with sweep
    * latest readings table
    * cluster probability synthesis
    * scenario start/stop/reset
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


EXPORT_COLUMNS = [
    "timestamp",
    "range_km",
    "velocity_mps",
    "azimuth_deg",
    "elevation_deg",
    "rcs_m2",
    "signal_strength_db",
    "target_type",
    "confidence",
    "gmm_cluster",
    "gmm_log_likelihood",
    "gmm_prob_cluster_0",
    "gmm_prob_cluster_1",
    "gmm_prob_cluster_2",
    "gmm_prob_cluster_3",
    "gmm_prob_cluster_4",
    "gmm_prob_cluster_5",
]


TARGET_TYPES = ["fighter", "airliner", "bomber", "drone", "tanker", "helicopter"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_choice(items, weights):
    total = sum(weights)
    if total <= 0:
        return random.choice(items)
    r = random.random() * total
    upto = 0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]


def dirichlet_like(k: int):
    vals = [random.gammavariate(1.0 + random.random() * 2.0, 1.0) for _ in range(k)]
    s = sum(vals)
    return [v / s for v in vals]


@dataclass
class Aircraft:
    ident: str
    target_type: str
    x_km: float
    y_km: float
    z_km: float
    vx_kmps: float
    vy_kmps: float
    vz_kmps: float
    rcs_m2: float

    def step(self, dt: float, max_range: float):
        self.x_km += self.vx_kmps * dt
        self.y_km += self.vy_kmps * dt
        self.z_km = max(0.1, self.z_km + self.vz_kmps * dt)

        # Bounce off the boundary to keep targets in view
        r = math.hypot(self.x_km, self.y_km)
        if r > max_range * 0.97:
            angle = math.atan2(self.y_km, self.x_km)
            inward = angle + math.pi + random.uniform(-0.4, 0.4)
            speed = math.hypot(self.vx_kmps, self.vy_kmps)
            self.vx_kmps = math.cos(inward) * speed
            self.vy_kmps = math.sin(inward) * speed
            self.x_km = math.cos(angle) * max_range * 0.9
            self.y_km = math.sin(angle) * max_range * 0.9

        # Mild maneuvering
        self.vx_kmps += random.uniform(-0.004, 0.004)
        self.vy_kmps += random.uniform(-0.004, 0.004)
        self.vz_kmps += random.uniform(-0.001, 0.001)

        # Limit speed
        planar_speed = math.hypot(self.vx_kmps, self.vy_kmps)
        max_speed = 0.42  # km/s ~= 420 m/s
        if planar_speed > max_speed:
            scale = max_speed / planar_speed
            self.vx_kmps *= scale
            self.vy_kmps *= scale


@dataclass
class RadarReading:
    timestamp: str
    range_km: float
    velocity_mps: float
    azimuth_deg: float
    elevation_deg: float
    rcs_m2: float
    signal_strength_db: float
    target_type: str
    confidence: float
    gmm_cluster: int
    gmm_log_likelihood: float
    gmm_probs: list[float] = field(default_factory=list)

    def as_row(self):
        row = {
            "timestamp": self.timestamp,
            "range_km": round(self.range_km, 2),
            "velocity_mps": round(self.velocity_mps, 2),
            "azimuth_deg": round(self.azimuth_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "rcs_m2": round(self.rcs_m2, 3),
            "signal_strength_db": round(self.signal_strength_db, 2),
            "target_type": self.target_type,
            "confidence": round(self.confidence, 4),
            "gmm_cluster": self.gmm_cluster,
            "gmm_log_likelihood": round(self.gmm_log_likelihood, 4),
        }
        for i in range(6):
            row[f"gmm_prob_cluster_{i}"] = round(self.gmm_probs[i], 6)
        return row


class RadarEngine:
    def __init__(self):
        self.max_range_km = 300.0
        self.aircraft: list[Aircraft] = []
        self.false_returns = []
        self.current_time = datetime(2026, 1, 1, 0, 0, 0)
        self.track_counter = 1

    def reset(self):
        self.aircraft.clear()
        self.false_returns.clear()
        self.current_time = datetime(2026, 1, 1, 0, 0, 0)
        self.track_counter = 1

    def populate_aircraft(self, count: int):
        self.aircraft.clear()
        for _ in range(count):
            ttype = weighted_choice(
                TARGET_TYPES,
                [2.2, 1.6, 0.7, 1.4, 0.9, 0.7],
            )
            angle = random.uniform(0, math.tau)
            radius = random.uniform(20, self.max_range_km * 0.92)
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            z = random.uniform(0.2, 12.0)
            speed = {
                "fighter": random.uniform(0.20, 0.36),
                "airliner": random.uniform(0.20, 0.27),
                "bomber": random.uniform(0.18, 0.25),
                "drone": random.uniform(0.04, 0.12),
                "tanker": random.uniform(0.16, 0.22),
                "helicopter": random.uniform(0.03, 0.09),
            }[ttype]
            heading = random.uniform(0, math.tau)
            vx = math.cos(heading) * speed
            vy = math.sin(heading) * speed
            vz = random.uniform(-0.01, 0.01)
            rcs = {
                "fighter": random.uniform(1.0, 5.0),
                "airliner": random.uniform(10.0, 40.0),
                "bomber": random.uniform(8.0, 25.0),
                "drone": random.uniform(0.05, 1.2),
                "tanker": random.uniform(12.0, 35.0),
                "helicopter": random.uniform(1.0, 8.0),
            }[ttype]
            self.aircraft.append(
                Aircraft(
                    ident=f"T{self.track_counter:03d}",
                    target_type=ttype,
                    x_km=x,
                    y_km=y,
                    z_km=z,
                    vx_kmps=vx,
                    vy_kmps=vy,
                    vz_kmps=vz,
                    rcs_m2=rcs,
                )
            )
            self.track_counter += 1

    def _cluster_for(self, target_type: str):
        mapping = {
            "fighter": 1,
            "airliner": 5,
            "bomber": 3,
            "drone": 4,
            "tanker": 2,
            "helicopter": 0,
            "clutter": random.randint(0, 5),
        }
        return mapping.get(target_type, random.randint(0, 5))


    @staticmethod
    def _format_timestamp(dt: datetime) -> str:
        # Cross-platform timestamp formatting.
        # Avoid %-m / %-d because they fail on some systems, especially Windows.
        return f"{dt.month}/{dt.day}/{dt.year} {dt:%H:%M:%S}"

    def step(
        self,
        dt_sec: float,
        weather_enabled: bool,
        weather_severity: float,
        jamming_enabled: bool,
        jamming_severity: float,
        clutter_enabled: bool,
    ):
        self.current_time += timedelta(seconds=dt_sec)
        for ac in self.aircraft:
            ac.step(dt_sec, self.max_range_km)

        readings: list[RadarReading] = []

        weather_penalty = weather_severity * 9.0 if weather_enabled else 0.0
        jam_penalty = jamming_severity * 14.0 if jamming_enabled else 0.0

        for ac in self.aircraft:
            rng = math.sqrt(ac.x_km ** 2 + ac.y_km ** 2 + ac.z_km ** 2)
            azimuth = (math.degrees(math.atan2(ac.y_km, ac.x_km)) + 360.0) % 360.0
            elevation = math.degrees(math.atan2(ac.z_km, max(0.001, math.hypot(ac.x_km, ac.y_km))))
            radial_vel_kmps = (ac.x_km * ac.vx_kmps + ac.y_km * ac.vy_kmps + ac.z_km * ac.vz_kmps) / max(rng, 0.001)

            free_space_loss = 20 * math.log10(max(rng, 1.0))
            base_signal = 60.0 + 5.5 * math.log10(max(ac.rcs_m2, 0.05)) - free_space_loss
            scintillation = random.uniform(-2.0, 2.0)
            signal_db = base_signal - weather_penalty - jam_penalty + scintillation

            confidence = 0.95 - (weather_severity * 0.25 if weather_enabled else 0.0) - (jamming_severity * 0.4 if jamming_enabled else 0.0)
            confidence -= clamp((rng / self.max_range_km) * 0.2, 0, 0.2)
            confidence += random.uniform(-0.06, 0.03)
            confidence = clamp(confidence, 0.05, 0.99)

            probs = dirichlet_like(6)
            cluster = self._cluster_for(ac.target_type)
            probs[cluster] += 2.0
            s = sum(probs)
            probs = [p / s for p in probs]
            log_like = -abs(random.gauss(2.4, 0.8)) - (1.0 - confidence) * 3.5

            readings.append(
                RadarReading(
                    timestamp=self._format_timestamp(self.current_time),
                    range_km=rng + random.uniform(-0.8, 0.8) * (weather_severity if weather_enabled else 0.12),
                    velocity_mps=radial_vel_kmps * 1000 + random.uniform(-8, 8),
                    azimuth_deg=azimuth + random.uniform(-0.7, 0.7) * (1 + (jamming_severity if jamming_enabled else 0)),
                    elevation_deg=elevation + random.uniform(-0.4, 0.4),
                    rcs_m2=ac.rcs_m2 * (1 + random.uniform(-0.05, 0.05)),
                    signal_strength_db=signal_db,
                    target_type=ac.target_type,
                    confidence=confidence,
                    gmm_cluster=cluster,
                    gmm_log_likelihood=log_like,
                    gmm_probs=probs,
                )
            )

        # Add clutter / weather cells / jammer ghosts
        extras = 0
        if clutter_enabled:
            extras += random.randint(0, 2)
        if weather_enabled:
            extras += random.randint(0, 2 + int(weather_severity * 4))
        if jamming_enabled:
            extras += random.randint(1, 2 + int(jamming_severity * 5))

        for _ in range(extras):
            rng = random.uniform(8, self.max_range_km)
            azimuth = random.uniform(0, 360)
            elevation = random.uniform(-1, 12)
            signal = random.uniform(-15, 18)
            if weather_enabled:
                signal += weather_severity * random.uniform(-8, 4)
            if jamming_enabled:
                signal += jamming_severity * random.uniform(-2, 10)
            confidence = random.uniform(0.05, 0.55 if jamming_enabled or weather_enabled else 0.35)
            probs = dirichlet_like(6)
            cluster = random.randint(0, 5)
            probs[cluster] += 1.6
            probs = [p / sum(probs) for p in probs]
            readings.append(
                RadarReading(
                    timestamp=self._format_timestamp(self.current_time),
                    range_km=rng,
                    velocity_mps=random.uniform(-120, 120),
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    rcs_m2=random.uniform(0.03, 3.5),
                    signal_strength_db=signal,
                    target_type="clutter",
                    confidence=confidence,
                    gmm_cluster=cluster,
                    gmm_log_likelihood=-abs(random.gauss(4.5, 1.6)),
                    gmm_probs=probs,
                )
            )

        return readings


class RadarSimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Radar Simulator")
        self.geometry("1450x920")
        self.minsize(1280, 820)
        self.configure(bg="#0b1220")

        self.engine = RadarEngine()
        self.running = False
        self.history: list[dict] = []
        self.latest_readings: list[RadarReading] = []
        self.sweep_angle = 0.0
        self.last_update = time.time()
        self.after_id = None

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._configure_styles()
        self._build_variables()
        self._build_ui()
        self._reseed_targets()
        self._schedule_refresh()

    def _configure_styles(self):
        self.style.configure("TFrame", background="#0b1220")
        self.style.configure("TLabelframe", background="#111827", foreground="#e5e7eb")
        self.style.configure("TLabelframe.Label", background="#111827", foreground="#e5e7eb")
        self.style.configure("TLabel", background="#0b1220", foreground="#dbe4ee")
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#f8fafc")
        self.style.configure("Sub.TLabel", foreground="#94a3b8")
        self.style.configure("TButton", padding=6)
        self.style.configure("TCheckbutton", background="#0b1220", foreground="#dbe4ee")
        self.style.configure("TRadiobutton", background="#0b1220", foreground="#dbe4ee")
        self.style.configure("Treeview", background="#0f172a", fieldbackground="#0f172a", foreground="#e2e8f0", rowheight=24)
        self.style.configure("Treeview.Heading", background="#1f2937", foreground="#f8fafc")
        self.style.map("TButton", background=[("active", "#374151")])

    def _build_variables(self):
        self.aircraft_count_var = tk.IntVar(value=10)
        self.weather_enabled_var = tk.BooleanVar(value=False)
        self.weather_severity_var = tk.DoubleVar(value=0.35)
        self.jamming_enabled_var = tk.BooleanVar(value=False)
        self.jamming_severity_var = tk.DoubleVar(value=0.40)
        self.clutter_enabled_var = tk.BooleanVar(value=True)
        self.tick_ms_var = tk.IntVar(value=350)
        self.batch_points_var = tk.IntVar(value=250)
        self.max_range_var = tk.DoubleVar(value=300.0)

        self.param_vars = {col: tk.BooleanVar(value=True) for col in EXPORT_COLUMNS}

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Advanced Radar Simulator", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            top,
            text="Live polar radar display, scenario controls, export tools, and simulated readings close to the uploaded schema.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body, width=430)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_radar_panel(left)
        self._build_bottom_panel(left)
        self._build_controls_panel(right)

    def _build_radar_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Radar Scope")
        frame.pack(fill="both", expand=True)

        self.radar_canvas = tk.Canvas(frame, bg="#06111f", highlightthickness=0)
        self.radar_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.radar_canvas.bind("<Configure>", lambda e: self._draw_radar())

    def _build_bottom_panel(self, parent):
        bottom = ttk.Frame(parent)
        bottom.pack(fill="x", pady=(8, 0))

        table_frame = ttk.LabelFrame(bottom, text="Latest Readings")
        table_frame.pack(side="left", fill="both", expand=True)

        columns = ("time", "type", "range", "azimuth", "velocity", "signal", "confidence", "cluster")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        for col, width in [
            ("time", 130),
            ("type", 90),
            ("range", 80),
            ("azimuth", 80),
            ("velocity", 80),
            ("signal", 80),
            ("confidence", 85),
            ("cluster", 70),
        ]:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)

        if HAS_MATPLOTLIB:
            chart_frame = ttk.LabelFrame(bottom, text="Live Signal / Confidence")
            chart_frame.pack(side="left", fill="both", expand=False, padx=(8, 0))
            chart_frame.configure(width=360)

            self.fig = Figure(figsize=(4.2, 2.8), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("Recent Mean Signal / Confidence")
            self.ax.set_xlabel("Sweep")
            self.ax.set_ylabel("Value")
            self.signal_series = []
            self.conf_series = []
            self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
            self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        else:
            fallback = ttk.LabelFrame(bottom, text="Charts")
            fallback.pack(side="left", fill="both", expand=False, padx=(8, 0))
            ttk.Label(
                fallback,
                text="matplotlib not found.\nThe simulator will still run,\nbut live charts are disabled.",
            ).pack(padx=18, pady=18)

    def _build_controls_panel(self, parent):
        scenario = ttk.LabelFrame(parent, text="Scenario Controls")
        scenario.pack(fill="x")

        grid = ttk.Frame(scenario)
        grid.pack(fill="x", padx=8, pady=8)

        ttk.Label(grid, text="Aircraft").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grid, from_=1, to=50, textvariable=self.aircraft_count_var, width=8).grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(grid, text="Tick (ms)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(grid, from_=100, to=2000, increment=50, textvariable=self.tick_ms_var, width=8).grid(row=0, column=3, sticky="ew", padx=6)

        ttk.Label(grid, text="Max Range (km)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(grid, from_=50, to=600, increment=10, textvariable=self.max_range_var, width=8).grid(row=1, column=1, sticky="ew", padx=6)

        ttk.Label(grid, text="Batch Rows").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(grid, from_=10, to=5000, increment=10, textvariable=self.batch_points_var, width=8).grid(row=1, column=3, sticky="ew", padx=6)

        ttk.Checkbutton(grid, text="Inclement Weather", variable=self.weather_enabled_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Scale(grid, from_=0.0, to=1.0, variable=self.weather_severity_var, orient="horizontal").grid(row=2, column=2, columnspan=2, sticky="ew", padx=6, pady=(6, 0))

        ttk.Checkbutton(grid, text="Jamming", variable=self.jamming_enabled_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Scale(grid, from_=0.0, to=1.0, variable=self.jamming_severity_var, orient="horizontal").grid(row=3, column=2, columnspan=2, sticky="ew", padx=6, pady=(6, 0))

        ttk.Checkbutton(grid, text="Background Clutter", variable=self.clutter_enabled_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        for i in range(4):
            grid.columnconfigure(i, weight=1)

        buttons = ttk.Frame(scenario)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Start", command=self.start).pack(side="left")
        ttk.Button(buttons, text="Stop", command=self.stop).pack(side="left", padx=6)
        ttk.Button(buttons, text="Reset", command=self.reset).pack(side="left")
        ttk.Button(buttons, text="Reseed Targets", command=self._reseed_targets).pack(side="left", padx=6)

        export = ttk.LabelFrame(parent, text="Export Columns")
        export.pack(fill="both", expand=True, pady=(8, 0))

        inner = ttk.Frame(export)
        inner.pack(fill="both", expand=True, padx=8, pady=8)

        for idx, col in enumerate(EXPORT_COLUMNS):
            ttk.Checkbutton(inner, text=col, variable=self.param_vars[col]).grid(row=idx // 2, column=idx % 2, sticky="w", padx=4, pady=2)

        export_buttons = ttk.Frame(export)
        export_buttons.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(export_buttons, text="Export Current Buffer", command=self.export_current_buffer).pack(side="left")
        ttk.Button(export_buttons, text="Generate Batch CSV", command=self.export_batch_csv).pack(side="left", padx=6)
        ttk.Button(export_buttons, text="Select All", command=lambda: self._set_all_params(True)).pack(side="left", padx=6)
        ttk.Button(export_buttons, text="Clear All", command=lambda: self._set_all_params(False)).pack(side="left")

        stats = ttk.LabelFrame(parent, text="Session Metrics")
        stats.pack(fill="x", pady=(8, 0))
        self.metrics_text = tk.Text(stats, height=10, bg="#08111f", fg="#dbeafe", insertbackground="white", relief="flat")
        self.metrics_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.metrics_text.insert("1.0", "Session metrics will appear here.\n")
        self.metrics_text.configure(state="disabled")

    def _set_all_params(self, value: bool):
        for var in self.param_vars.values():
            var.set(value)

    def _reseed_targets(self):
        self.engine.reset()
        self.engine.max_range_km = self.max_range_var.get()
        self.engine.populate_aircraft(self.aircraft_count_var.get())
        self.history.clear()
        self.latest_readings.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._draw_radar()
        self._update_metrics()

    def start(self):
        if not self.running:
            self.running = True

    def stop(self):
        self.running = False

    def reset(self):
        self.running = False
        self._reseed_targets()

    def _schedule_refresh(self):
        delay = max(50, int(self.tick_ms_var.get()))
        self.after_id = self.after(delay, self._tick)

    def _tick(self):
        try:
            if self.running:
                self._simulate_step()
            self._draw_radar()
            self._schedule_refresh()
        except Exception as exc:
            messagebox.showerror("Simulator Error", str(exc))
            self.running = False

    def _simulate_step(self):
        dt_sec = max(0.1, self.tick_ms_var.get() / 1000.0)
        self.engine.max_range_km = self.max_range_var.get()

        readings = self.engine.step(
            dt_sec=dt_sec,
            weather_enabled=self.weather_enabled_var.get(),
            weather_severity=self.weather_severity_var.get(),
            jamming_enabled=self.jamming_enabled_var.get(),
            jamming_severity=self.jamming_severity_var.get(),
            clutter_enabled=self.clutter_enabled_var.get(),
        )
        self.latest_readings = readings
        self._append_history(readings)
        self._refresh_table()
        self._update_metrics()
        self._update_chart()
        self.sweep_angle = (self.sweep_angle + dt_sec * 120) % 360

    def _append_history(self, readings: list[RadarReading]):
        for r in readings:
            self.history.append(r.as_row())
        max_keep = 10000
        if len(self.history) > max_keep:
            self.history = self.history[-max_keep:]

    def _refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in sorted(self.latest_readings, key=lambda x: x.range_km)[:18]:
            self.tree.insert(
                "",
                "end",
                values=(
                    r.timestamp.split()[-1],
                    r.target_type,
                    f"{r.range_km:.1f}",
                    f"{r.azimuth_deg:.1f}",
                    f"{r.velocity_mps:.0f}",
                    f"{r.signal_strength_db:.1f}",
                    f"{r.confidence:.2f}",
                    r.gmm_cluster,
                ),
            )

    def _update_metrics(self):
        text = []
        actual = [r for r in self.latest_readings if r.target_type != "clutter"]
        clutter = [r for r in self.latest_readings if r.target_type == "clutter"]
        text.append(f"Tracks configured: {len(self.engine.aircraft)}")
        text.append(f"Returns this sweep: {len(self.latest_readings)}")
        text.append(f"Aircraft returns: {len(actual)}")
        text.append(f"Clutter / ghost returns: {len(clutter)}")
        if self.latest_readings:
            text.append(f"Mean signal: {statistics.fmean(r.signal_strength_db for r in self.latest_readings):.2f} dB")
            text.append(f"Mean confidence: {statistics.fmean(r.confidence for r in self.latest_readings):.3f}")
            text.append(f"Closest return: {min(r.range_km for r in self.latest_readings):.2f} km")
            text.append(f"Farthest return: {max(r.range_km for r in self.latest_readings):.2f} km")
        text.append(f"Weather: {'ON' if self.weather_enabled_var.get() else 'OFF'} ({self.weather_severity_var.get():.2f})")
        text.append(f"Jamming: {'ON' if self.jamming_enabled_var.get() else 'OFF'} ({self.jamming_severity_var.get():.2f})")
        text.append(f"Clutter: {'ON' if self.clutter_enabled_var.get() else 'OFF'}")
        text.append(f"Buffered rows: {len(self.history)}")

        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(text))
        self.metrics_text.configure(state="disabled")

    def _update_chart(self):
        if not HAS_MATPLOTLIB or not self.latest_readings:
            return
        self.signal_series.append(statistics.fmean(r.signal_strength_db for r in self.latest_readings))
        self.conf_series.append(statistics.fmean(r.confidence for r in self.latest_readings))
        self.signal_series = self.signal_series[-40:]
        self.conf_series = self.conf_series[-40:]

        self.ax.clear()
        self.ax.plot(self.signal_series, label="Signal (dB)")
        self.ax.plot(self.conf_series, label="Confidence")
        self.ax.set_title("Recent Mean Signal / Confidence")
        self.ax.set_xlabel("Sweep")
        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.chart_canvas.draw_idle()

    def _draw_radar(self):
        canvas = self.radar_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 400)
        h = max(canvas.winfo_height(), 400)
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.45

        # Background rings
        for frac in [0.25, 0.5, 0.75, 1.0]:
            r = radius * frac
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#123f2a", width=2)
            label_rng = self.engine.max_range_km * frac
            canvas.create_text(cx + 6, cy - r + 10, text=f"{label_rng:.0f} km", fill="#66ff99", anchor="w", font=("Consolas", 10))

        # Crosshairs
        canvas.create_line(cx - radius, cy, cx + radius, cy, fill="#164e2b", width=2)
        canvas.create_line(cx, cy - radius, cx, cy + radius, fill="#164e2b", width=2)

        # Sweep line + sector glow
        sweep_rad = math.radians(self.sweep_angle - 90)
        for tail in range(8):
            a = math.radians(self.sweep_angle - tail * 3 - 90)
            alpha = max(30, 170 - tail * 18)
            color = f"#{0:02x}{alpha:02x}{80:02x}"
            x = cx + math.cos(a) * radius
            y = cy + math.sin(a) * radius
            canvas.create_line(cx, cy, x, y, fill=color, width=max(1, 6 - tail))
        sx = cx + math.cos(sweep_rad) * radius
        sy = cy + math.sin(sweep_rad) * radius
        canvas.create_line(cx, cy, sx, sy, fill="#7cffb2", width=3)

        # Weather cloud
        if self.weather_enabled_var.get():
            severity = self.weather_severity_var.get()
            for _ in range(int(12 + severity * 35)):
                ang = random.uniform(0, math.tau)
                rr = random.uniform(radius * 0.18, radius * (0.25 + severity * 0.5))
                x = cx + math.cos(ang) * rr + random.uniform(-18, 18)
                y = cy + math.sin(ang) * rr + random.uniform(-18, 18)
                size = random.uniform(4, 18)
                canvas.create_oval(x - size, y - size, x + size, y + size, outline="", fill="#203d56")

        # Jamming wedge
        if self.jamming_enabled_var.get():
            jam = self.jamming_severity_var.get()
            jam_angle = (self.sweep_angle + 60) % 360
            for width in [20, 30, 40]:
                extent = 12 + jam * width
                canvas.create_arc(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    start=jam_angle,
                    extent=extent,
                    style="arc",
                    outline="#ff7b72",
                    width=2,
                )

        # Blips
        for r in self.latest_readings:
            pr = (r.range_km / self.engine.max_range_km) * radius
            ang = math.radians(r.azimuth_deg - 90)
            x = cx + math.cos(ang) * pr
            y = cy + math.sin(ang) * pr

            if r.target_type == "clutter":
                color = "#f59e0b"
                size = 2 + r.confidence * 3
            else:
                color = "#7cffb2" if r.confidence > 0.6 else "#93c5fd"
                size = 3 + max(0, r.signal_strength_db + 20) / 15

            canvas.create_oval(x - size, y - size, x + size, y + size, outline="", fill=color)
            if r.target_type != "clutter":
                canvas.create_text(x + 8, y - 8, text=r.target_type[:3].upper(), fill="#d1fae5", anchor="w", font=("Consolas", 9))

        canvas.create_text(
            12,
            12,
            text=f"SWEEP {self.sweep_angle:06.2f}°",
            fill="#a7f3d0",
            anchor="nw",
            font=("Consolas", 12, "bold"),
        )

    def _selected_columns(self):
        cols = [c for c, v in self.param_vars.items() if v.get()]
        return cols or EXPORT_COLUMNS[:]

    def export_current_buffer(self):
        if not self.history:
            messagebox.showinfo("No Data", "No readings are buffered yet. Start the simulation first.")
            return
        cols = self._selected_columns()
        path = filedialog.asksaveasfilename(
            title="Save current radar buffer",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="advanced_radar_buffer.csv",
        )
        if not path:
            return
        self._write_csv(path, self.history, cols)
        messagebox.showinfo("Export Complete", f"Saved {len(self.history)} rows to:\n{path}")

    def export_batch_csv(self):
        cols = self._selected_columns()
        count = max(1, self.batch_points_var.get())
        path = filedialog.asksaveasfilename(
            title="Save generated radar dataset",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="advanced_simulated_radar_readings.csv",
        )
        if not path:
            return

        saved_running = self.running
        self.running = False

        # Snapshot config
        if not self.engine.aircraft:
            self.engine.populate_aircraft(self.aircraft_count_var.get())

        rows = []
        for _ in range(count):
            readings = self.engine.step(
                dt_sec=max(0.1, self.tick_ms_var.get() / 1000.0),
                weather_enabled=self.weather_enabled_var.get(),
                weather_severity=self.weather_severity_var.get(),
                jamming_enabled=self.jamming_enabled_var.get(),
                jamming_severity=self.jamming_severity_var.get(),
                clutter_enabled=self.clutter_enabled_var.get(),
            )
            rows.extend([r.as_row() for r in readings])

        self._write_csv(path, rows, cols)
        self.running = saved_running
        messagebox.showinfo("Batch Export Complete", f"Saved {len(rows)} rows to:\n{path}")

    @staticmethod
    def _write_csv(path: str, rows: list[dict], columns: list[str]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in columns})


def main():
    app = RadarSimulatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()

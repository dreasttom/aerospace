import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


class HypersonicAirflowApp:
    """
    Educational hypersonic boundary-layer / heating / shock demo.

    This is a concept visualizer for Mach 5+ flight, not a CFD solver.
    It estimates compressibility-driven changes in transition tendency,
    stagnation heating, and a stylized detached bow shock / boundary layer.
    """

    G0 = 9.80665
    R = 287.05
    GAMMA = 1.4

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hypersonic Airflow Transition Demo")
        self.root.geometry("1360x800")
        self.root.minsize(1180, 720)

        self.anim = None
        self.phase = 0.0
        self.stream_paths = []
        self.particles = []

        self._build_variables()
        self._build_layout()
        self._build_plot()
        self.run_simulation()

    def _build_variables(self):
        self.vars = {
            "altitude_m": tk.DoubleVar(value=25000.0),
            "mach": tk.DoubleVar(value=6.0),
            "length_m": tk.DoubleVar(value=2.5),
            "nose_radius_m": tk.DoubleVar(value=0.15),
            "aoa_deg": tk.DoubleVar(value=5.0),
            "wall_temp_k": tk.DoubleVar(value=800.0),
            "roughness_um": tk.DoubleVar(value=40.0),
            "turbulence_pct": tk.DoubleVar(value=0.25),
            "pressure_grad": tk.DoubleVar(value=0.15),
            "surface_catalytic": tk.DoubleVar(value=0.5),
        }

        self.out = {
            "temp": tk.StringVar(value="-"),
            "rho": tk.StringVar(value="-"),
            "mu": tk.StringVar(value="-"),
            "a": tk.StringVar(value="-"),
            "speed": tk.StringVar(value="-"),
            "re_l": tk.StringVar(value="-"),
            "re_crit": tk.StringVar(value="-"),
            "x_tr": tk.StringVar(value="-"),
            "t0": tk.StringVar(value="-"),
            "t_aw": tk.StringVar(value="-"),
            "qdot": tk.StringVar(value="-"),
            "shock": tk.StringVar(value="-"),
            "state": tk.StringVar(value="-"),
            "notes": tk.StringVar(value="Ready."),
        }

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 12))

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(
            left,
            text="Hypersonic Flow Explorer (Mach 5+)",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            left,
            text=(
                "Adjust hypersonic flight and thermal parameters, then press Run.\n"
                "The tool estimates transition tendency, heating, and shock geometry."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        inputs = ttk.LabelFrame(left, text="Inputs", padding=10)
        inputs.pack(fill="x")

        self._add_slider(inputs, "Altitude (m)", "altitude_m", 10000, 50000, 250, 0)
        self._add_slider(inputs, "Mach number", "mach", 5.0, 15.0, 0.1, 1)
        self._add_slider(inputs, "Vehicle length (m)", "length_m", 0.5, 12.0, 0.1, 2)
        self._add_slider(inputs, "Nose radius (m)", "nose_radius_m", 0.02, 1.0, 0.01, 3)
        self._add_slider(inputs, "Angle of attack (deg)", "aoa_deg", -2.0, 20.0, 0.25, 4)
        self._add_slider(inputs, "Wall temperature (K)", "wall_temp_k", 250.0, 2500.0, 10.0, 5)
        self._add_slider(inputs, "Surface roughness (μm)", "roughness_um", 0.0, 500.0, 1.0, 6)
        self._add_slider(inputs, "Free-stream turbulence (%)", "turbulence_pct", 0.01, 5.0, 0.01, 7)
        self._add_slider(inputs, "Pressure-gradient factor", "pressure_grad", -1.0, 1.0, 0.01, 8)
        self._add_slider(inputs, "Catalytic surface factor", "surface_catalytic", 0.0, 1.0, 0.01, 9)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Run", command=self.run_simulation).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Reset", command=self.reset_defaults).pack(side="left", fill="x", expand=True, padx=(8, 0))

        outputs = ttk.LabelFrame(left, text="Calculated Results", padding=10)
        outputs.pack(fill="x", pady=(4, 0))

        rows = [
            ("Static temperature T (K)", "temp"),
            ("Air density ρ (kg/m³)", "rho"),
            ("Dynamic viscosity μ (Pa·s)", "mu"),
            ("Speed of sound a (m/s)", "a"),
            ("Vehicle speed V (m/s)", "speed"),
            ("Length Reynolds number Re_L", "re_l"),
            ("Estimated critical Re_x", "re_crit"),
            ("Transition location x/L", "x_tr"),
            ("Stagnation temperature T0 (K)", "t0"),
            ("Adiabatic wall temperature Taw (K)", "t_aw"),
            ("Stagnation heating proxy q̇ (W/cm²)", "qdot"),
            ("Detached shock stand-off (m)", "shock"),
            ("Boundary-layer state", "state"),
        ]

        for i, (label, key) in enumerate(rows):
            ttk.Label(outputs, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(outputs, textvariable=self.out[key], font=("Consolas", 10)).grid(
                row=i, column=1, sticky="e", padx=(10, 0), pady=2
            )

        ttk.Label(
            left,
            textvariable=self.out["notes"],
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

        notes = ttk.LabelFrame(left, text="Model Notes", padding=10)
        notes.pack(fill="x", pady=(10, 0))
        ttk.Label(
            notes,
            text=(
                "This version is aimed at hypersonic intuition: compressibility, heating,\n"
                "and shock-layer behavior dominate. Chemistry, dissociation, ionization,\n"
                "surface ablation, and real-gas effects are represented only very crudely."
            ),
            justify="left",
        ).pack(anchor="w")

        plot_frame = ttk.LabelFrame(right, text="Hypersonic Visual Demo", padding=8)
        plot_frame.pack(fill="both", expand=True)
        self.plot_container = plot_frame

    def _add_slider(self, parent, label, key, min_val, max_val, step, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        scale = tk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            orient="horizontal",
            resolution=step,
            variable=self.vars[key],
            length=260,
        )
        scale.grid(row=row, column=1, sticky="ew", padx=(10, 0))
        parent.columnconfigure(1, weight=1)

    def _build_plot(self):
        self.fig = Figure(figsize=(8.6, 6.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-0.6, 3.7)
        self.ax.set_ylim(-1.25, 1.25)
        self.ax.set_xlabel("x / vehicle length")
        self.ax.set_ylabel("y / vehicle length")
        self.ax.set_title("Hypersonic Shock Layer and Transition Visualization")
        self.ax.grid(True, alpha=0.25)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    @staticmethod
    def isa_properties(altitude_m: float):
        # Simplified ISA to 50 km.
        h = max(0.0, min(50000.0, altitude_m))

        if h <= 11000.0:
            T = 288.15 - 0.0065 * h
            P = 101325.0 * (T / 288.15) ** 5.255877
        elif h <= 20000.0:
            T = 216.65
            P = 22632.06 * math.exp(-HypersonicAirflowApp.G0 * (h - 11000.0) / (HypersonicAirflowApp.R * T))
        elif h <= 32000.0:
            T = 216.65 + 0.001 * (h - 20000.0)
            P20 = 5474.89
            P = P20 * (216.65 / T) ** (HypersonicAirflowApp.G0 / (0.001 * HypersonicAirflowApp.R))
        else:
            T = 228.65 + 0.0028 * (h - 32000.0)
            P32 = 868.02
            P = P32 * (228.65 / T) ** (HypersonicAirflowApp.G0 / (0.0028 * HypersonicAirflowApp.R))

        rho = P / (HypersonicAirflowApp.R * T)
        mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
        a = math.sqrt(HypersonicAirflowApp.GAMMA * HypersonicAirflowApp.R * T)
        return T, P, rho, mu, a

    def compute_case(self):
        altitude = self.vars["altitude_m"].get()
        mach = max(5.0, self.vars["mach"].get())
        length = max(0.1, self.vars["length_m"].get())
        nose_radius = max(0.005, self.vars["nose_radius_m"].get())
        aoa_deg = self.vars["aoa_deg"].get()
        wall_temp = max(150.0, self.vars["wall_temp_k"].get())
        roughness_um = max(0.0, self.vars["roughness_um"].get())
        turbulence_pct = max(0.01, self.vars["turbulence_pct"].get())
        pressure_grad = max(-1.0, min(1.0, self.vars["pressure_grad"].get()))
        catalytic = max(0.0, min(1.0, self.vars["surface_catalytic"].get()))

        T, P, rho, mu, a = self.isa_properties(altitude)
        V = mach * a
        re_l = rho * V * length / mu

        # Hypersonic temperature quantities.
        T0 = T * (1.0 + 0.5 * (self.GAMMA - 1.0) * mach ** 2)
        recovery_factor = 0.88 if mach < 8 else 0.92
        Taw = T + recovery_factor * (T0 - T)
        Taw *= (1.0 + 0.08 * catalytic)

        # Crude Fay-Riddell-inspired heating proxy, scaled to W/cm^2 for display.
        q_w_m2 = 1.83e-4 * math.sqrt(max(rho, 1e-8) / max(nose_radius, 1e-5)) * V ** 3
        q_w_m2 *= (1.0 + 0.12 * catalytic)
        q_w_cm2 = q_w_m2 / 10000.0

        # Hypersonic transition estimate:
        # start from higher Re_x,crit but reduce strongly with roughness, turbulence,
        # wall cooling/heating mismatch, Mach level, AoA, and adverse gradient.
        wall_ratio = wall_temp / max(Taw, 1.0)
        re_crit = 1.8e6
        re_crit *= 1.0 - min(0.75, roughness_um / 650.0)
        re_crit *= 1.0 - min(0.70, 0.20 * math.log10(1.0 + 100.0 * turbulence_pct))
        re_crit *= 1.0 - min(0.45, max(0.0, mach - 5.0) / 18.0)
        re_crit *= 1.0 - min(0.30, abs(aoa_deg) / 35.0)
        re_crit *= 1.0 + 0.35 * min(0.6, max(-0.6, -pressure_grad))
        re_crit *= 1.0 - 0.25 * max(0.0, pressure_grad)
        re_crit *= 1.0 + 0.22 * max(-0.5, min(0.5, 1.0 - wall_ratio))
        re_crit = max(1.0e5, min(4.0e6, re_crit))

        re_per_m = rho * V / mu
        x_tr = re_crit / max(1.0, re_per_m)
        x_over_l = x_tr / length

        if x_over_l >= 1.0:
            state = "Mostly laminar over vehicle length"
            x_over_l = 1.0
        elif x_over_l <= 0.03:
            state = "Rapidly turbulent after nose region"
            x_over_l = 0.03
        else:
            state = "Laminar-to-turbulent along body"

        # Detached bow shock stand-off proxy.
        shock_standoff = nose_radius * (0.78 / max(0.25, mach - 1.0)) * (1.0 + 0.02 * abs(aoa_deg))
        shock_standoff = max(0.01 * nose_radius, shock_standoff)

        notes = []
        if mach >= 10:
            notes.append("Very high Mach intensifies heating and amplifies real-gas uncertainty.")
        if q_w_cm2 > 80:
            notes.append("Heating proxy is severe; TPS effects would matter strongly.")
        if roughness_um > 120:
            notes.append("Surface roughness strongly promotes early transition in hypersonic flow.")
        if wall_ratio < 0.6:
            notes.append("Relatively cool wall tends to alter instability behavior and heating response.")
        if pressure_grad > 0.2:
            notes.append("Adverse pressure gradient promotes earlier transition.")
        elif pressure_grad < -0.2:
            notes.append("Favorable pressure gradient may delay transition somewhat.")
        if not notes:
            notes.append("This case is moderate for a hypersonic educational demo.")

        return {
            "T": T,
            "P": P,
            "rho": rho,
            "mu": mu,
            "a": a,
            "V": V,
            "mach": mach,
            "re_l": re_l,
            "re_crit": re_crit,
            "x_over_l": x_over_l,
            "T0": T0,
            "Taw": Taw,
            "q_w_cm2": q_w_cm2,
            "shock_standoff": shock_standoff,
            "state": state,
            "aoa_deg": aoa_deg,
            "length": length,
            "nose_radius": nose_radius,
            "notes": " ".join(notes),
        }

    def _vehicle_shape(self, aoa_rad):
        x = np.linspace(0.0, 3.0, 500)
        r = np.zeros_like(x)

        nose = x <= 0.5
        body = (x > 0.5) & (x <= 2.4)
        tail = x > 2.4

        # Blunt nose + slender body + slight boat-tail.
        r[nose] = 0.28 * np.sqrt(np.clip(1.0 - ((x[nose] - 0.5) / 0.5) ** 2, 0.0, 1.0))
        r[body] = 0.28 - 0.03 * (x[body] - 0.5)
        r[tail] = 0.223 - 0.14 * (x[tail] - 2.4)
        r = np.clip(r, 0.01, None)

        upper = np.vstack([x, r])
        lower = np.vstack([x, -r])

        rot = np.array([
            [math.cos(aoa_rad), -math.sin(aoa_rad)],
            [math.sin(aoa_rad),  math.cos(aoa_rad)],
        ])

        upper = rot @ upper
        lower = rot @ lower
        return upper, lower

    def draw_case(self, result):
        self.ax.clear()
        self.ax.set_xlim(-0.6, 3.7)
        self.ax.set_ylim(-1.25, 1.25)
        self.ax.set_xlabel("x / vehicle length")
        self.ax.set_ylabel("y / vehicle length")
        self.ax.set_title("Hypersonic Shock Layer and Transition Visualization")
        self.ax.grid(True, alpha=0.25)

        aoa_rad = math.radians(result["aoa_deg"])
        xtr = result["x_over_l"] * 3.0
        shock_offset = min(0.55, 3.0 * result["shock_standoff"] / max(result["length"], 1e-6))
        heating_scale = min(1.0, result["q_w_cm2"] / 120.0)

        upper, lower = self._vehicle_shape(aoa_rad)
        self.ax.fill(
            np.r_[upper[0], lower[0][::-1]],
            np.r_[upper[1], lower[1][::-1]],
            alpha=0.28,
        )

        x_center = np.linspace(0.0, 3.0, 500)
        nose_profile = 0.34 * np.sqrt(np.clip(1.0 - ((x_center - 0.55) / 0.58) ** 2, 0.0, 1.0))
        body_profile = np.interp(x_center, upper[0], upper[1], left=upper[1][0], right=upper[1][-1])
        shock_y = np.maximum(np.abs(body_profile) + shock_offset + 0.05 * np.exp(-((x_center - 0.4) / 0.45) ** 2), nose_profile + shock_offset)

        rot = np.array([
            [math.cos(aoa_rad), -math.sin(aoa_rad)],
            [math.sin(aoa_rad),  math.cos(aoa_rad)],
        ])
        shock_upper = rot @ np.vstack([x_center, shock_y])
        shock_lower = rot @ np.vstack([x_center, -shock_y])
        self.ax.plot(shock_upper[0], shock_upper[1], linestyle="--", linewidth=1.6)
        self.ax.plot(shock_lower[0], shock_lower[1], linestyle="--", linewidth=1.6)

        trans_point = rot @ np.array([[xtr], [0.0]])
        self.ax.axvline(trans_point[0, 0], linestyle=":", linewidth=1.7)
        self.ax.text(
            trans_point[0, 0] + 0.03,
            1.07,
            f"Transition\n x/L = {result['x_over_l']:.2f}",
            fontsize=10,
            va="top",
        )

        self.ax.text(
            shock_upper[0][25],
            shock_upper[1][25] + 0.08,
            "Detached bow shock",
            fontsize=10,
        )

        # Heating band near stagnation / nose.
        hx = np.linspace(0.0, 0.55, 120)
        hy = 0.10 + 0.16 * np.exp(-((hx - 0.1) / 0.2) ** 2) * (0.4 + heating_scale)
        hot_upper = rot @ np.vstack([hx, hy])
        hot_lower = rot @ np.vstack([hx, -hy])
        self.ax.fill(
            np.r_[hot_upper[0], hot_lower[0][::-1]],
            np.r_[hot_upper[1], hot_lower[1][::-1]],
            alpha=0.15 + 0.18 * heating_scale,
        )

        self.stream_paths.clear()
        self.particles.clear()

        y0s = np.linspace(-0.95, 0.95, 13)
        xs = np.linspace(-0.45, 3.45, 500)
        shock_x0 = -shock_offset * 0.7

        for y0 in y0s:
            deflection = 0.0
            ys = np.full_like(xs, y0)

            # Upstream compression and shock turning.
            shock_bump = 0.22 * np.exp(-((xs - 0.08) / 0.18) ** 2)
            ys += np.sign(y0 if y0 != 0 else 1.0) * shock_bump * (0.35 + 0.65 * math.exp(-abs(y0)))

            # Near-body streamline bending.
            body_pull = 0.16 * np.exp(-((xs - 0.8) / 0.9) ** 2)
            ys += np.sign(y0 if y0 != 0 else 1.0) * body_pull * (0.20 + 0.35 * math.exp(-abs(y0)))

            # Turbulent waviness after transition.
            turb_mask = xs >= xtr
            waviness = np.zeros_like(xs)
            waviness[turb_mask] = 0.015 * np.sin(28 * xs[turb_mask] + 8 * y0)
            ys += waviness

            self.ax.plot(xs[xs <= xtr], ys[xs <= xtr], linewidth=1.2, alpha=0.85)
            self.ax.plot(xs[xs >= xtr], ys[xs >= xtr], linewidth=1.2, alpha=0.85)

            particle, = self.ax.plot([], [], marker="o", markersize=4, linestyle="None")
            self.stream_paths.append((xs, ys))
            self.particles.append(particle)

        textbox = (
            f"Mach = {result['mach']:.1f}\n"
            f"Re_L = {result['re_l']:.2e}\n"
            f"T0 = {result['T0']:.0f} K\n"
            f"q̇ ≈ {result['q_w_cm2']:.1f} W/cm²\n"
            f"State: {result['state']}"
        )
        self.ax.text(
            2.55,
            1.1,
            textbox,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
        )

        self.canvas.draw_idle()
        self.start_animation(result["mach"])

    def start_animation(self, mach):
        if self.anim is not None:
            self.anim.event_source.stop()

        speed_scale = max(0.5, min(2.8, mach / 5.0))

        def update(_frame):
            self.phase = (self.phase + 0.012 * speed_scale) % 1.0
            for i, ((xs, ys), particle) in enumerate(zip(self.stream_paths, self.particles)):
                frac = (self.phase + i * 0.065) % 1.0
                idx = min(len(xs) - 1, int(frac * (len(xs) - 1)))
                particle.set_data([xs[idx]], [ys[idx]])
            return self.particles

        self.anim = FuncAnimation(self.fig, update, interval=35, blit=True)
        self.canvas.draw_idle()

    def run_simulation(self):
        try:
            result = self.compute_case()
        except Exception as exc:
            messagebox.showerror("Calculation Error", str(exc))
            return

        self.out["temp"].set(f"{result['T']:.1f}")
        self.out["rho"].set(f"{result['rho']:.4e}")
        self.out["mu"].set(f"{result['mu']:.3e}")
        self.out["a"].set(f"{result['a']:.1f}")
        self.out["speed"].set(f"{result['V']:.1f}")
        self.out["re_l"].set(f"{result['re_l']:.3e}")
        self.out["re_crit"].set(f"{result['re_crit']:.3e}")
        self.out["x_tr"].set(f"{result['x_over_l']:.3f}")
        self.out["t0"].set(f"{result['T0']:.1f}")
        self.out["t_aw"].set(f"{result['Taw']:.1f}")
        self.out["qdot"].set(f"{result['q_w_cm2']:.2f}")
        self.out["shock"].set(f"{result['shock_standoff']:.4f}")
        self.out["state"].set(result["state"])
        self.out["notes"].set(result["notes"])

        self.draw_case(result)

    def reset_defaults(self):
        defaults = {
            "altitude_m": 25000.0,
            "mach": 6.0,
            "length_m": 2.5,
            "nose_radius_m": 0.15,
            "aoa_deg": 5.0,
            "wall_temp_k": 800.0,
            "roughness_um": 40.0,
            "turbulence_pct": 0.25,
            "pressure_grad": 0.15,
            "surface_catalytic": 0.5,
        }
        for key, value in defaults.items():
            self.vars[key].set(value)
        self.run_simulation()


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    app = HypersonicAirflowApp(root)
    root.mainloop()

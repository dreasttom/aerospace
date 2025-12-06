"""
hypersonic_wedge_visualizer_advanced.py
NOTE THIS IS INCOMPLETE INTENTIONALLY. IT RUNS BUT DOES NOT PUT THE DATA IN THE GRAPH
THAT IS LEFT AS AN EXCERCISE FOR THE STUDENT
Educational script to visualize hypersonic airflow over a 2D wedge.

FEATURES
--------
1) Single-case visualization:
   - User chooses:
       * Freestream Mach number M_inf
       * Wedge half-angle theta (deg)
       * Freestream static temperature T_inf (K)
   - Script:
       * Solves the θ–β–M relation for oblique shock angle β
       * Computes basic oblique-shock property ratios
       * Plots:
           (a) Geometry: wedge + attached oblique shock + flow arrows
           (b) Temperature map above the wedge (qualitative visualization)
           (c) θ–β curve for that Mach, marking the chosen operating point

2) Mach sweep visualization:
   - User chooses a set of Mach numbers (e.g. 3, 5, 7, 10)
   - Script plots β vs θ curves for each Mach on the same figure.

ASSUMPTIONS
-----------
- Perfect gas, gamma = 1.4
- 2D, inviscid, adiabatic flow
- Oblique shock attached to a sharp wedge (when possible)
- Hypersonic or at least supersonic flow (M > 1)
"""

import math
import matplotlib.pyplot as plt

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

GAMMA = 1.4  # Ratio of specific heats for air (assumed constant)


# ============================================================
# CORE OBLIQUE SHOCK RELATIONS
# ============================================================

def theta_beta_m_relation(beta, M, theta, gamma=GAMMA):
    """
    Oblique shock θ–β–M relation written as f(beta) = 0.

    Classical relation:
        tan(theta) = 2 * cot(beta) * (M^2 * sin^2(beta) - 1)
                     / [ M^2 (gamma + cos(2beta)) + 2 ]

    We define:
        f(beta) = tan(theta) - RHS

    So, f(beta) = 0 should give the physical shock angle.

    Parameters
    ----------
    beta : float
        Shock angle in radians.
    M : float
        Upstream Mach number (dimensionless).
    theta : float
        Flow deflection angle (radians) = wedge half-angle (for a wedge).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        f(beta) value (we want f(beta) = 0).
    """
    # Avoid singular or invalid beta values.
    if beta <= 0.0 or beta >= math.pi / 2:
        return 1e9

    tan_theta = math.tan(theta)
    sin_beta = math.sin(beta)
    cos_beta = math.cos(beta)
    cos_2beta = math.cos(2.0 * beta)

    # cot(beta) = cos(beta) / sin(beta)
    cot_beta = cos_beta / sin_beta

    M2 = M * M
    sin2_beta = sin_beta * sin_beta

    numerator = 2.0 * cot_beta * (M2 * sin2_beta - 1.0)
    denominator = M2 * (gamma + cos_2beta) + 2.0

    rhs = numerator / denominator

    return tan_theta - rhs


def solve_shock_angle(M, theta_deg, gamma=GAMMA):
    """
    Solve for the *weak* oblique shock angle β (deg) for a given M and θ (deg).

    We use a simple bisection method between:
      beta_low  = theta
      beta_high = ~90° (π/2 rad)

    If there is no sign change in f(beta), we assume there is
    no attached oblique shock solution (shock becomes detached)
    for that combination of M and θ.

    Parameters
    ----------
    M : float
        Freestream Mach number.
    theta_deg : float
        Flow deflection / wedge half-angle (degrees).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    beta_deg : float or None
        Weak solution shock angle in degrees, or None if no solution.
    """
    theta = math.radians(theta_deg)

    beta_low = theta
    beta_high = math.pi / 2

    f_low = theta_beta_m_relation(beta_low, M, theta, gamma)
    f_high = theta_beta_m_relation(beta_high * 0.999, M, theta, gamma)

    # If no sign change, bisection won't work -> probably no attached shock
    if f_low * f_high > 0:
        return None

    for _ in range(100):
        beta_mid = 0.5 * (beta_low + beta_high)
        f_mid = theta_beta_m_relation(beta_mid, M, theta, gamma)

        if abs(f_mid) < 1e-7:
            break

        if f_low * f_mid < 0:
            beta_high = beta_mid
            f_high = f_mid
        else:
            beta_low = beta_mid
            f_low = f_mid

    beta_deg = math.degrees(beta_mid)
    return beta_deg


def oblique_shock_relations(M1, beta_deg, theta_deg, gamma=GAMMA):
    """
    Compute basic oblique-shock relations for a given M1, beta, and theta.

    We treat the oblique shock as a normal shock in the normal direction.

    Steps:
      - M_n1 = M1 * sin(beta)
      - Use normal shock relations to get p2/p1, rho2/rho1, T2/T1, M_n2
      - Convert M_n2 back to M2 using M2 = M_n2 / sin(beta - theta)

    Parameters
    ----------
    M1 : float
        Upstream Mach number.
    beta_deg : float
        Shock angle (degrees).
    theta_deg : float
        Flow deflection angle = wedge half-angle (degrees).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    dict
        Keys: "M_n1", "M_n2", "M2", "p2_p1", "rho2_rho1", "T2_T1"
    """
    beta = math.radians(beta_deg)
    theta = math.radians(theta_deg)

    M_n1 = M1 * math.sin(beta)
    M_n1_sq = M_n1 * M_n1

    # Normal shock pressure ratio
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M_n1_sq - 1.0)

    # Density ratio
    rho2_rho1 = ((gamma + 1.0) * M_n1_sq) / (2.0 + (gamma - 1.0) * M_n1_sq)

    # Temperature ratio
    T2_T1 = p2_p1 / rho2_rho1

    # Downstream normal Mach
    numerator = 1.0 + 0.5 * (gamma - 1.0) * M_n1_sq
    denominator = gamma * M_n1_sq - 0.5 * (gamma - 1.0)
    M_n2_sq = numerator / denominator
    M_n2 = math.sqrt(M_n2_sq)

    # Convert back to flow direction
    M2 = M_n2 / math.sin(beta - theta)

    return {
        "M_n1": M_n1,
        "M_n2": M_n2,
        "M2": M2,
        "p2_p1": p2_p1,
        "rho2_rho1": rho2_rho1,
        "T2_T1": T2_T1
    }


def stagnation_temperature(T, M, gamma=GAMMA):
    """
    Stagnation temperature relation:

        T0 = T * [1 + (gamma - 1)/2 * M^2]

    Parameters
    ----------
    T : float
        Static temperature (K).
    M : float
        Mach number.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Stagnation temperature T0 (K).
    """
    return T * (1.0 + 0.5 * (gamma - 1.0) * M * M)


# ============================================================
# INPUT HELPERS
# ============================================================

def input_with_default(prompt, default, cast_type=float):
    """
    Prompt the user for input with a default value.

    If the user just presses ENTER, the default is used.

    Parameters
    ----------
    prompt : str
        Prompt string.
    default : any
        Default value if user presses ENTER.
    cast_type : callable
        Function used to convert the string to desired type (e.g. float, int).

    Returns
    -------
    value : same type as default (after casting).
    """
    full_prompt = f"{prompt} [default = {default}]: "
    s = input(full_prompt)

    if s.strip() == "":
        return default

    try:
        return cast_type(s)
    except ValueError:
        print("Invalid input. Using default value.")
        return default


# ============================================================
# PLOTTING: WEDGE + SHOCK
# ============================================================

def plot_wedge_and_shock(M_inf, theta_deg, beta_deg, shock_info, T_inf):
    """
    2D geometric plot showing:
      - Wedge surface
      - Attached oblique shock
      - Incoming flow arrows
      - Text box with key numbers (M_inf, θ, β, p2/p1, T2/T1, etc.)

    This gives a simple "cartoon" visualization of hypersonic flow.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    theta_deg : float
        Wedge half-angle (deg).
    beta_deg : float
        Shock angle (deg).
    shock_info : dict
        Output from oblique_shock_relations().
    T_inf : float
        Freestream static temperature (K).
    """
    theta = math.radians(theta_deg)
    beta = math.radians(beta_deg)

    # Simple geometry setup
    x_max = 1.0  # arbitrary length scale
    wedge_x = [0.0, x_max]
    wedge_y = [0.0, x_max * math.tan(theta)]

    shock_x = [0.0, x_max]
    shock_y = [0.0, x_max * math.tan(beta)]

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # Plot wedge and centerline
    plt.plot(wedge_x, wedge_y, linewidth=3, label="Wedge surface")
    plt.axhline(0.0, linestyle="--", linewidth=1, label="Flow direction")

    # Plot shock
    plt.plot(shock_x, shock_y, linestyle="--", linewidth=2, label="Oblique shock")

    # Flow arrows to the left
    for y_arrow in [0.0, 0.1, -0.1]:
        plt.arrow(-0.3, y_arrow, 0.25, 0.0,
                  head_width=0.02, head_length=0.05,
                  length_includes_head=True)

    # Labels
    plt.text(0.6, wedge_y[1] + 0.03, f"Wedge, θ = {theta_deg:.1f}°", fontsize=10)
    plt.text(0.6, shock_y[1] + 0.03, f"Shock, β ≈ {beta_deg:.1f}°", fontsize=10)

    # Temperatures
    T0_inf = stagnation_temperature(T_inf, M_inf)
    T2_T1 = shock_info["T2_T1"]
    T2 = T2_T1 * T_inf
    M2 = shock_info["M2"]
    T0_2 = stagnation_temperature(T2, M2)

    # Text box
    text_x = -0.95
    text_y = 0.45
    props = dict(boxstyle="round", facecolor="white", alpha=0.9)

    info_text = (
        f"M_inf = {M_inf:.2f}\n"
        f"θ = {theta_deg:.1f}°   β ≈ {beta_deg:.1f}°\n"
        f"p2/p1 ≈ {shock_info['p2_p1']:.2f}\n"
        f"T2/T1 ≈ {shock_info['T2_T1']:.2f}\n"
        f"M2 ≈ {shock_info['M2']:.2f}\n"
        f"T_inf = {T_inf:.1f} K   T0_inf ≈ {T0_inf:.1f} K\n"
        f"T2 ≈ {T2:.1f} K   T0_2 ≈ {T0_2:.1f} K"
    )

    plt.text(text_x, text_y, info_text, fontsize=9, bbox=props)

    plt.xlabel("x (arbitrary units)")
    plt.ylabel("y (arbitrary units)")
    plt.title("Hypersonic Flow over a 2D Wedge\n(Geometry + Shock)")

    plt.xlim(-1.0, 1.2)
    plt.ylim(-0.5, 0.8)
    plt.grid(True, linestyle=":")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # Return T2 so we can reuse it in temperature map
    return T2


# ============================================================
# PLOTTING: TEMPERATURE MAP (QUALITATIVE)
# ============================================================

def plot_temperature_map(theta_deg, T_inf, T2):
    """
    Create a qualitative 2D "temperature map" near the wedge.

    This is NOT a CFD solution, just a cartoon:
      - Above the wedge, near the surface, temperature is high (~T2).
      - Farther away from the surface, temperature relaxes back toward T_inf.

    Implementation:
      - We define a 2D grid in x (0 to 1) and y (0 to 0.3).
      - For each point above the wedge surface y_wedge(x) = x * tan(theta),
        we assign T by a simple linear decay from T2 at the surface
        to T_inf at some distance y_max above the surface.

    Parameters
    ----------
    theta_deg : float
        Wedge angle (deg).
    T_inf : float
        Freestream static temperature (K).
    T2 : float
        Post-shock static temperature just above the surface (K).
    """
    theta = math.radians(theta_deg)

    # Grid resolution
    nx = 300
    ny = 150

    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 0.3

    xs = [x_min + (x_max - x_min) * i / (nx - 1) for i in range(nx)]
    ys = [y_min + (y_max - y_min) * j / (ny - 1) for j in range(ny)]

    # Temperature field
    T_field = [[T_inf for _ in range(nx)] for _ in range(ny)]

    # For each grid point, determine distance above wedge
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            # Wedge surface height at this x
            y_wedge = x * math.tan(theta)

            if y >= y_wedge:
                # Distance above wedge surface
                dy = y - y_wedge

                # We assume temperature goes from T2 at dy = 0
                # to T_inf at dy = (y_max - y_min)
                d_max = y_max - y_min
                frac = min(dy / d_max, 1.0)

                T_here = T2 - (T2 - T_inf) * frac
                T_field[j][i] = T_here
            else:
                # Below wedge surface: we can either mark as solid
                # or just leave as T_inf. Here we leave T_inf.
                T_field[j][i] = T_inf

    # Convert to a 2D array for imshow
    import numpy as np
    T_array = np.array(T_field)

    plt.figure(figsize=(6, 4))
    # imshow expects [row, col] indexing, which we have (j,i)
    # extent sets coordinate axes in x and y
    plt.imshow(T_array,
               origin="lower",
               extent=[x_min, x_max, y_min, y_max],
               aspect="auto")
    plt.colorbar(label="Temperature (K)")
    plt.xlabel("x (arbitrary units)")
    plt.ylabel("y (arbitrary units)")
    plt.title("Qualitative Temperature Map above Wedge Surface")

    # Plot the wedge line on top for reference
    wedge_x = [0.0, x_max]
    wedge_y = [0.0, x_max * math.tan(theta)]
    plt.plot(wedge_x, wedge_y, color="black", linewidth=2)

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOTTING: SHOCK ANGLE vs WEDGE ANGLE FOR FIXED MACH
# ============================================================

def compute_theta_beta_curve_for_M(M, theta_max_deg=40.0, dtheta_deg=0.5):
    """
    Compute arrays theta_list, beta_list for a given Mach M by
    scanning wedge angles from small values up to theta_max_deg.

    For each theta, we solve for beta (weak shock). If there is
    no attached solution (solve_shock_angle returns None),
    we stop the scan.

    Parameters
    ----------
    M : float
        Mach number.
    theta_max_deg : float
        Upper bound of wedge angles to scan (deg).
    dtheta_deg : float
        Step in wedge angle (deg).

    Returns
    -------
    theta_list : list of float
        Wedge angles (deg) where an attached shock was found.
    beta_list : list of float
        Corresponding shock angles (deg).
    """
    theta_list = []
    beta_list = []

    theta_deg = dtheta_deg
    while theta_deg <= theta_max_deg:
        beta_deg = solve_shock_angle(M, theta_deg, gamma=GAMMA)
        if beta_deg is None:
            # No attached solution; stop here
            break

        theta_list.append(theta_deg)
        beta_list.append(beta_deg)
        theta_deg += dtheta_deg

    return theta_list, beta_list


def plot_theta_beta_for_M(M, theta_current_deg=None, beta_current_deg=None):
    """
    Plot shock angle β vs wedge angle θ for a given Mach number M.

    Optionally, mark a specific (theta_current, beta_current) point
    corresponding to the user's single-case choice.

    Parameters
    ----------
    M : float
        Mach number.
    theta_current_deg : float or None
        Wedge angle (deg) for the current operating point.
    beta_current_deg : float or None
        Shock angle (deg) for the current operating point.
    """
    theta_list, beta_list = compute_theta_beta_curve_for_M(M)

    if len(theta_list) == 0:
        print(f"\nNo attached-shock θ–β curve could be computed for M = {M:.2f}.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(theta_list, beta_list, label=f"M = {M:.2f}")

    if theta_current_deg is not None and beta_current_deg is not None:
        plt.scatter([theta_current_deg], [beta_current_deg],
                    color="red", zorder=5,
                    label="Current operating point")
        plt.text(theta_current_deg + 0.5, beta_current_deg,
                 f"({theta_current_deg:.1f}°, {beta_current_deg:.1f}°)",
                 fontsize=9)

    plt.xlabel("Wedge angle θ (deg)")
    plt.ylabel("Shock angle β (deg)")
    plt.title(f"Shock Angle vs Wedge Angle for M = {M:.2f}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PLOTTING: MACH SWEEP (MULTIPLE β–θ CURVES)
# ============================================================

def plot_theta_beta_mach_sweep(M_list, theta_max_deg=40.0, dtheta_deg=0.5):
    """
    Plot θ–β curves for a list of Mach numbers on the same figure.

    Parameters
    ----------
    M_list : list of float
        Mach numbers to include (e.g. [3, 5, 7, 10]).
    theta_max_deg : float
        Upper bound of wedge angles to scan (deg).
    dtheta_deg : float
        Step in wedge angle (deg).
    """
    plt.figure(figsize=(7, 5))

    for M in M_list:
        theta_list, beta_list = compute_theta_beta_curve_for_M(
            M, theta_max_deg=theta_max_deg, dtheta_deg=dtheta_deg
        )

        if len(theta_list) == 0:
            # No curve for this Mach
            print(f"  (No attached-shock θ–β curve for M = {M:.2f})")
            continue

        plt.plot(theta_list, beta_list, label=f"M = {M:.2f}")

    plt.xlabel("Wedge angle θ (deg)")
    plt.ylabel("Shock angle β (deg)")
    plt.title("Shock Angle β vs Wedge Angle θ\nfor Multiple Mach Numbers")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MENU OPTION 1: SINGLE CASE (ALL VISUALS)
# ============================================================

def run_single_case():
    """
    Menu option 1:
      - Ask user for M_inf, θ, T_inf.
      - Solve for β and shock properties.
      - Plot:
          (1) Wedge + shock geometry + text
          (2) Temperature map above wedge
          (3) θ–β curve for that Mach with operating point marked
    """
    print("\n=== Hypersonic Wedge Flow Case ===\n")
    print("Assumptions:")
    print("  - Perfect gas, gamma = 1.4")
    print("  - 2D, inviscid, adiabatic flow")
    print("  - Sharp 2D wedge with attached oblique shock (when possible)\n")

    M_inf = input_with_default("Enter freestream Mach number M_inf", 7.0, float)
    theta_deg = input_with_default("Enter wedge half-angle θ (deg)", 10.0, float)
    T_inf = input_with_default("Enter freestream static temperature T_inf (K)", 220.0, float)

    if M_inf <= 1.0:
        print("\nWARNING: M_inf ≤ 1 is not really supersonic; shocks may not form as expected.\n")

    if theta_deg <= 0.0:
        print("\nERROR: Wedge angle must be > 0.\n")
        return

    # Solve θ–β–M for shock angle
    beta_deg = solve_shock_angle(M_inf, theta_deg, gamma=GAMMA)
    if beta_deg is None:
        print("\nNo attached oblique shock solution found (shock likely detached).")
        print("Try a smaller wedge angle or higher Mach.\n")
        return

    # Shock properties
    shock_info = oblique_shock_relations(M_inf, beta_deg, theta_deg, gamma=GAMMA)

    print("\n--- Flow / Shock Info ---")
    print(f"M_inf              = {M_inf:.3f}")
    print(f"θ (wedge)          = {theta_deg:.3f} deg")
    print(f"β (shock)          ≈ {beta_deg:.3f} deg")
    print(f"p2/p1              ≈ {shock_info['p2_p1']:.3f}")
    print(f"T2/T1              ≈ {shock_info['T2_T1']:.3f}")
    print(f"M2 (behind shock)  ≈ {shock_info['M2']:.3f}")
    print("--------------------------\n")

    # (1) Geometry + basic text
    T2 = plot_wedge_and_shock(M_inf, theta_deg, beta_deg, shock_info, T_inf)

    # (2) Temperature map
    plot_temperature_map(theta_deg, T_inf, T2)

    # (3) θ–β curve for this Mach (mark current point)
    plot_theta_beta_for_M(M_inf, theta_current_deg=theta_deg, beta_current_deg=beta_deg)


# ============================================================
# MENU OPTION 2: MACH SWEEP (β–θ CURVES)
# ============================================================

def run_mach_sweep():
    """
    Menu option 2:
      - Ask user for a comma-separated list of Mach numbers
        (e.g. "3,5,7,10").
      - Plot θ–β curves for each Mach on same figure.
    """
    print("\n=== Mach Number Sweep: θ–β Curves ===\n")
    print("You can see how the shock angle β changes with wedge angle θ")
    print("for multiple Mach numbers on the same plot.\n")

    default_M_list_str = "3,5,7,10"
    M_list_str = input_with_default(
        "Enter Mach numbers as comma-separated list (e.g. 3,5,7,10)",
        default_M_list_str,
        str
    )

    # Parse into floats
    M_list = []
    for token in M_list_str.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            M_val = float(token)
            if M_val > 1.0:
                M_list.append(M_val)
            else:
                print(f"  Skipping M = {M_val} (must be > 1)")
        except ValueError:
            print(f"  Could not parse '{token}' as a Mach number; skipping.")

    if len(M_list) == 0:
        print("\nNo valid Mach numbers. Aborting Mach sweep.\n")
        return

    theta_max_deg = input_with_default("Max wedge angle θ_max to plot (deg)", 40.0, float)
    dtheta_deg = input_with_default("Angle step Δθ for curves (deg)", 0.5, float)

    plot_theta_beta_mach_sweep(M_list, theta_max_deg=theta_max_deg, dtheta_deg=dtheta_deg)


# ============================================================
# MAIN MENU
# ============================================================

def main():
    """
    MAIN FUNCTION

    Menu:
      1) Single-case hypersonic wedge visualization
         -> geometry, shock, temperature map, θ–β curve for that Mach
      2) Mach sweep
         -> multiple θ–β curves on one plot
      3) Quit
    """
    while True:
        print("===================================================")
        print("       Hypersonic Wedge Airflow Visualizer         ")
        print("===================================================")
        print("1) Single-case visualization (wedge, shock, T-map, θ–β)")
        print("2) Mach sweep (multiple θ–β curves)")
        print("3) Quit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            run_single_case()
        elif choice == "2":
            run_mach_sweep()
        elif choice == "3":
            print("\nExiting hypersonic airflow visualizer. Goodbye!\n")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()

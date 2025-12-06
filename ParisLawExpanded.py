"""
paris_law_menu_logged.py

Educational script to demonstrate Paris' Law for fatigue crack growth:

    da/dN = C * (ΔK)^m

With:
    ΔK = Y(a) * Δσ * sqrt(pi * a)

This version includes:
    - Text-based menu for interaction.
    - Simple constant-geometry model (Y = constant).
    - More realistic edge-crack model with Y(a/W).
    - Numerical integration of Paris' law.
    - Analytic solution for constant Y (where possible).
    - Logging of results to a CSV file.
    - An option to compare BOTH MODELS on the same plot
      for the same set of parameters.

Units (by convention here):
    - Crack lengths in mm (converted internally to meters).
    - Plate width W in mm (converted internally to meters).
    - Stress range in MPa (converted internally to Pa).
    - Output crack lengths for plotting are in mm.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

# ================================================================
# LOGGING SETUP
# ================================================================

LOG_FILE = "paris_law_results.csv"

# Define a fixed order of CSV columns to keep the file tidy
CSV_FIELDNAMES = [
    "timestamp",
    "model",
    "C",
    "m",
    "delta_sigma_MPa",
    "Y0",
    "W_mm",
    "a_initial_mm",
    "a_final_mm",
    "num_steps",
    "N_analytic",
    "N_numeric"
]


def log_results(model_name,
                C,
                m,
                delta_sigma_MPa,
                Y0,
                W_mm,
                a_initial_mm,
                a_final_mm,
                num_steps,
                N_analytic,
                N_numeric,
                filename=LOG_FILE):
    """
    Append a row of results to the CSV log file.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. "constant_Y" or "edge_crack" or "comparison_constant_Y".
    C, m : float
        Paris law constants.
    delta_sigma_MPa : float
        Stress range in MPa (just for easier human reading in the log).
    Y0 : float or None
        Constant geometry factor (for the simple model). Use None or '' if not applicable.
    W_mm : float or None
        Plate width in mm (for edge crack model). Use None or '' if not applicable.
    a_initial_mm, a_final_mm : float
        Initial and final crack sizes in mm.
    num_steps : int
        Number of numerical steps used.
    N_analytic : float or None
        Analytic number of cycles (if available). Use None if not applicable.
    N_numeric : float
        Numerically integrated number of cycles.
    filename : str
        Path to CSV log file.
    """
    file_exists = os.path.isfile(filename)

    # Prepare row dictionary (convert None to '' for CSV friendliness)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "C": C,
        "m": m,
        "delta_sigma_MPa": delta_sigma_MPa,
        "Y0": "" if Y0 is None else Y0,
        "W_mm": "" if W_mm is None else W_mm,
        "a_initial_mm": a_initial_mm,
        "a_final_mm": a_final_mm,
        "num_steps": num_steps,
        "N_analytic": "" if N_analytic is None else N_analytic,
        "N_numeric": N_numeric
    }

    # Open file in append mode and write header if it's a new file
    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ================================================================
# GEOMETRY FACTORS (Y FUNCTIONS)
# ================================================================

def Y_constant(a, Y0=1.0):
    """
    Geometry factor that is simply constant (no dependence on a).

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m). *Not used*, kept only for interface consistency.
    Y0 : float
        Constant geometry factor.

    Returns
    -------
    float or np.ndarray
        Geometry factor Y = Y0.
    """
    return Y0


def Y_edge_crack(a, W):
    """
    Geometry factor Y(a) for an edge crack in a finite-width plate:

        Y(a) ≈ 1.12 - 0.23*α + 10.55*α^2 - 21.72*α^3 + 30.39*α^4

    where α = a / W (a = crack length, W = plate width).

    Typically used for a/W ≲ 0.6.

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m).
    W : float
        Plate width (m).

    Returns
    -------
    float or np.ndarray
        Geometry factor Y(a).
    """
    alpha = a / W
    return 1.12 - 0.23 * alpha + 10.55 * alpha**2 - 21.72 * alpha**3 + 30.39 * alpha**4


# ================================================================
# CORE PARIS LAW FUNCTIONS
# ================================================================

def delta_K(a, delta_sigma, Y_function):
    """
    Compute the stress intensity factor range ΔK for a given crack length
    and geometry factor function Y(a).

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m).
    delta_sigma : float
        Stress range Δσ (Pa).
    Y_function : callable
        Function of a that returns Y(a).

    Returns
    -------
    float or np.ndarray
        Stress intensity factor range ΔK (Pa * sqrt(m)).
    """
    Y_val = Y_function(a)
    return Y_val * delta_sigma * np.sqrt(np.pi * a)


def paris_law_da_dN(a, C, m, delta_sigma, Y_function):
    """
    Compute da/dN from Paris' law:

        da/dN = C * (ΔK)^m

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m).
    C : float
        Paris law constant.
    m : float
        Paris law exponent.
    delta_sigma : float
        Stress range Δσ (Pa).
    Y_function : callable
        Function of a that returns Y(a).

    Returns
    -------
    float or np.ndarray
        Crack growth rate da/dN (m per cycle).
    """
    dK = delta_K(a, delta_sigma, Y_function)
    return C * (dK ** m)


def paris_analytic_cycles_constant_Y(a_initial, a_final, C, m, delta_sigma, Y0):
    """
    Analytical solution for the number of cycles required to grow a crack
    from a_initial to a_final with Paris' law, assuming CONSTANT Y = Y0.

    da/dN = C * (Y0 * Δσ * sqrt(pi * a))^m
          = C * (Y0 * Δσ * sqrt(pi))^m * a^(m/2)

    dN = da / [ C * (Y0 * Δσ * sqrt(pi))^m * a^(m/2) ]

    If m ≠ 2:
        N = [1 / ( C * (Y0 * Δσ * sqrt(pi))^m )] *
            [ 1 / (1 - m/2) ] *
            ( a_final^(1 - m/2) - a_initial^(1 - m/2) )

    If m = 2:
        N = [1 / ( C * (Y0 * Δσ * sqrt(pi))^2 )] * ln(a_final / a_initial)

    Parameters
    ----------
    a_initial, a_final : float
        Crack lengths (m).
    C, m : float
        Paris law constants.
    delta_sigma : float
        Stress range Δσ (Pa).
    Y0 : float
        Constant geometry factor.

    Returns
    -------
    float
        Number of cycles to grow from a_initial to a_final.
    """
    factor = (Y0 * delta_sigma * np.sqrt(np.pi)) ** m

    if np.isclose(m, 2.0):
        N = (1.0 / (C * factor)) * np.log(a_final / a_initial)
    else:
        exponent = 1.0 - m / 2.0
        N = (1.0 / (C * factor)) * (1.0 / exponent) * \
            (a_final ** exponent - a_initial ** exponent)

    return N


def paris_numerical_growth(a_initial, a_final, C, m, delta_sigma,
                           Y_function, num_steps=1000):
    """
    Numerical integration of Paris' law to obtain crack length vs. cycles.

    Approach (explicit Euler in 'a'):
        - Divide [a_initial, a_final] into many small increments da.
        - For each step:
              dN = da / (da/dN)
          where da/dN is from Paris' law at the *previous* crack length.

    Parameters
    ----------
    a_initial, a_final : float
        Initial and final crack lengths (m).
    C, m : float
        Paris law constants.
    delta_sigma : float
        Stress range Δσ (Pa).
    Y_function : callable
        Function of a that returns Y(a).
    num_steps : int
        Number of crack-length steps.

    Returns
    -------
    N_values : np.ndarray
        Cycle count corresponding to each crack length step.
    a_values : np.ndarray
        Crack length values from a_initial to a_final.
    """
    a_values = np.linspace(a_initial, a_final, num_steps)
    N_values = np.zeros_like(a_values)

    for i in range(1, num_steps):
        a_prev = a_values[i - 1]
        a_curr = a_values[i]
        da = a_curr - a_prev

        da_dN_prev = paris_law_da_dN(a_prev, C, m, delta_sigma, Y_function)

        if da_dN_prev <= 0.0:
            dN = 0.0
        else:
            dN = da / da_dN_prev

        N_values[i] = N_values[i - 1] + dN

    return N_values, a_values


# ================================================================
# HELPER: INPUT WITH DEFAULT
# ================================================================

def input_with_default(prompt, default, cast_type=float):
    """
    Ask the user for input, but use a default if they just press Enter.

    Parameters
    ----------
    prompt : str
        Text shown to user.
    default : any
        Default value.
    cast_type : callable
        Function to cast the input string (e.g., float, int, str).

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


# ================================================================
# MODEL RUNNERS
# ================================================================

def run_simple_constant_Y_model():
    """
    Run the simple model:
        - Constant Y0.
        - Analytic solution for N.
        - Numerical integration.
        - Logging to CSV.
    """
    print("\n=== SIMPLE MODEL: CONSTANT Y ===\n")
    print("Units:")
    print("  - Crack lengths in mm (converted to meters internally).")
    print("  - Stress range in MPa (converted to Pa internally).\n")

    C = input_with_default("Enter Paris law constant C", 1e-12, float)
    m = input_with_default("Enter Paris law exponent m", 3.0, float)

    delta_sigma_MPa = input_with_default("Enter stress range Δσ (MPa)", 100.0, float)
    delta_sigma = delta_sigma_MPa * 1e6  # MPa -> Pa

    Y0 = input_with_default("Enter constant geometry factor Y0", 1.0, float)

    a_initial_mm = input_with_default("Enter initial crack size a_initial (mm)", 1.0, float)
    a_final_mm = input_with_default("Enter final crack size a_final (mm)", 10.0, float)

    a_initial = a_initial_mm / 1000.0  # mm -> m
    a_final = a_final_mm / 1000.0      # mm -> m

    num_steps = int(input_with_default("Number of steps for numerical integration",
                                       1000, int))

    # Analytic solution
    N_analytic = paris_analytic_cycles_constant_Y(a_initial, a_final, C, m,
                                                  delta_sigma, Y0)

    print("\nParis Law Analytical Solution (Constant Y)")
    print("-----------------------------------------")
    print(f"Initial crack size a_i = {a_initial:.6e} m")
    print(f"Final crack size   a_f = {a_final:.6e} m")
    print(f"Stress range Δσ         = {delta_sigma_MPa:.2f} MPa")
    print(f"Y0 (constant)           = {Y0:.3f}")
    print(f"Material constants: C = {C:.3e}, m = {m:.2f}")
    print(f"Analytical cycles N     = {N_analytic:.3e} cycles\n")

    # Numerical integration
    Y_func = lambda a: Y_constant(a, Y0=Y0)
    N_values, a_values = paris_numerical_growth(a_initial, a_final, C, m,
                                                delta_sigma, Y_func,
                                                num_steps=num_steps)

    N_numeric = N_values[-1]

    print("Paris Law Numerical Integration (Constant Y)")
    print("--------------------------------------------")
    print(f"Numerical estimate of cycles to reach a_f: "
          f"N = {N_numeric:.3e} cycles")
    print("(Compare this with the analytical solution above.)\n")

    # Log results
    log_results(
        model_name="constant_Y",
        C=C,
        m=m,
        delta_sigma_MPa=delta_sigma_MPa,
        Y0=Y0,
        W_mm=None,
        a_initial_mm=a_initial_mm,
        a_final_mm=a_final_mm,
        num_steps=num_steps,
        N_analytic=N_analytic,
        N_numeric=N_numeric
    )
    print(f"Results logged to '{LOG_FILE}'.\n")

    # Plot (a in mm)
    plt.figure()
    plt.plot(N_values, a_values * 1000.0)
    plt.xlabel("Number of Cycles, N")
    plt.ylabel("Crack Length, a (mm)")
    plt.title("Fatigue Crack Growth (Constant Y)\nCrack Length vs. Number of Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_edge_crack_model():
    """
    Run the edge-crack model:
        - Y(a/W) polynomial.
        - Numerical integration only.
        - Logging to CSV.
    """
    print("\n=== EDGE-CRACK MODEL: Y(a/W) ===\n")
    print("Geometry: Edge crack in a finite-width plate.")
    print("Approximate geometry factor:")
    print("  Y(a) ≈ 1.12 - 0.23*(a/W) + 10.55*(a/W)^2")
    print("           - 21.72*(a/W)^3 + 30.39*(a/W)^4\n")
    print("Units:")
    print("  - a_initial, a_final, W in mm (converted to meters).")
    print("  - Stress range in MPa (converted to Pa).\n")

    C = input_with_default("Enter Paris law constant C", 1e-12, float)
    m = input_with_default("Enter Paris law exponent m", 3.0, float)

    delta_sigma_MPa = input_with_default("Enter stress range Δσ (MPa)", 100.0, float)
    delta_sigma = delta_sigma_MPa * 1e6  # MPa -> Pa

    W_mm = input_with_default("Enter plate width W (mm)", 100.0, float)
    a_initial_mm = input_with_default("Enter initial crack size a_initial (mm)", 1.0, float)
    a_final_mm = input_with_default("Enter final crack size a_final (mm)", 10.0, float)

    W = W_mm / 1000.0
    a_initial = a_initial_mm / 1000.0
    a_final = a_final_mm / 1000.0

    num_steps = int(input_with_default("Number of steps for numerical integration",
                                       1000, int))

    # Warn if a/W is large (out of common validity range)
    if (a_final / W) > 0.6:
        print("\nWARNING: a_final / W > 0.6. The Y(a/W) approximation may be less accurate.\n")

    Y_func = lambda a: Y_edge_crack(a, W=W)

    N_values, a_values = paris_numerical_growth(a_initial, a_final, C, m,
                                                delta_sigma, Y_func,
                                                num_steps=num_steps)

    N_numeric = N_values[-1]

    print("\nParis Law Numerical Integration (Edge Crack)")
    print("--------------------------------------------")
    print(f"Plate width W           = {W_mm:.2f} mm")
    print(f"Initial crack size a_i  = {a_initial_mm:.2f} mm")
    print(f"Final crack size   a_f  = {a_final_mm:.2f} mm")
    print(f"Stress range Δσ         = {delta_sigma_MPa:.2f} MPa")
    print(f"Material constants: C = {C:.3e}, m = {m:.2f}")
    print(f"Numerical cycles N      = {N_numeric:.3e} cycles\n")

    # Log results (no analytic solution here)
    log_results(
        model_name="edge_crack",
        C=C,
        m=m,
        delta_sigma_MPa=delta_sigma_MPa,
        Y0=None,
        W_mm=W_mm,
        a_initial_mm=a_initial_mm,
        a_final_mm=a_final_mm,
        num_steps=num_steps,
        N_analytic=None,
        N_numeric=N_numeric
    )
    print(f"Results logged to '{LOG_FILE}'.\n")

    # Plot
    plt.figure()
    plt.plot(N_values, a_values * 1000.0)
    plt.xlabel("Number of Cycles, N")
    plt.ylabel("Crack Length, a (mm)")
    plt.title("Fatigue Crack Growth (Edge Crack Model)\nCrack Length vs. Number of Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_comparison_both_models():
    """
    Run both models with the SAME parameters (where applicable) and
    show them on the SAME PLOT.

    This:
        - Uses the same C, m, Δσ, a_initial, a_final, num_steps.
        - Requires both Y0 (for constant Y) and W_mm (for edge crack).
        - Computes:
            * Analytic and numeric for constant Y.
            * Numeric for edge crack.
        - Logs BOTH results to CSV.
        - Plots crack growth curves for both models on the same axes.
    """
    print("\n=== COMPARISON: CONSTANT Y vs. EDGE CRACK ===\n")
    print("This will:")
    print("  - Use ONE set of material/loading/crack parameters.")
    print("  - Apply them to BOTH models.")
    print("  - Plot crack length vs. cycles for both on the same plot.")
    print("  - Log BOTH sets of results.\n")

    C = input_with_default("Enter Paris law constant C", 1e-12, float)
    m = input_with_default("Enter Paris law exponent m", 3.0, float)

    delta_sigma_MPa = input_with_default("Enter stress range Δσ (MPa)", 100.0, float)
    delta_sigma = delta_sigma_MPa * 1e6  # MPa -> Pa

    Y0 = input_with_default("Enter constant geometry factor Y0 (for simple model)", 1.0, float)

    W_mm = input_with_default("Enter plate width W (mm) for edge crack model", 100.0, float)
    a_initial_mm = input_with_default("Enter initial crack size a_initial (mm)", 1.0, float)
    a_final_mm = input_with_default("Enter final crack size a_final (mm)", 10.0, float)

    a_initial = a_initial_mm / 1000.0
    a_final = a_final_mm / 1000.0
    W = W_mm / 1000.0

    num_steps = int(input_with_default("Number of steps for numerical integration", 1000, int))

    # Warn if a/W is large for edge crack model
    if (a_final / W) > 0.6:
        print("\nWARNING: a_final / W > 0.6. The Y(a/W) approximation may be less accurate.\n")

    # ------------------------------
    # CONSTANT Y MODEL
    # ------------------------------
    Y_const_func = lambda a: Y_constant(a, Y0=Y0)

    # Analytic
    N_analytic_const = paris_analytic_cycles_constant_Y(
        a_initial, a_final, C, m, delta_sigma, Y0
    )

    # Numerical
    N_const_values, a_const_values = paris_numerical_growth(
        a_initial, a_final, C, m, delta_sigma, Y_const_func, num_steps=num_steps
    )
    N_numeric_const = N_const_values[-1]

    # Log constant Y result
    log_results(
        model_name="comparison_constant_Y",
        C=C,
        m=m,
        delta_sigma_MPa=delta_sigma_MPa,
        Y0=Y0,
        W_mm=W_mm,  # log W as well for completeness
        a_initial_mm=a_initial_mm,
        a_final_mm=a_final_mm,
        num_steps=num_steps,
        N_analytic=N_analytic_const,
        N_numeric=N_numeric_const
    )

    # ------------------------------
    # EDGE CRACK MODEL
    # ------------------------------
    Y_edge_func = lambda a: Y_edge_crack(a, W=W)

    N_edge_values, a_edge_values = paris_numerical_growth(
        a_initial, a_final, C, m, delta_sigma, Y_edge_func, num_steps=num_steps
    )
    N_numeric_edge = N_edge_values[-1]

    # Log edge-crack result
    log_results(
        model_name="comparison_edge_crack",
        C=C,
        m=m,
        delta_sigma_MPa=delta_sigma_MPa,
        Y0=Y0,
        W_mm=W_mm,
        a_initial_mm=a_initial_mm,
        a_final_mm=a_final_mm,
        num_steps=num_steps,
        N_analytic=None,  # no analytic solution in this model
        N_numeric=N_numeric_edge
    )

    print("\nComparison Summary")
    print("------------------")
    print(f"Material constants: C = {C:.3e}, m = {m:.2f}")
    print(f"Stress range Δσ         = {delta_sigma_MPa:.2f} MPa")
    print(f"Initial crack size a_i  = {a_initial_mm:.2f} mm")
    print(f"Final crack size a_f    = {a_final_mm:.2f} mm")
    print(f"Plate width W           = {W_mm:.2f} mm")
    print(f"Constant Y0 (simple)    = {Y0:.3f}")
    print("")
    print(f"Constant Y model:")
    print(f"  Analytic N   = {N_analytic_const:.3e} cycles")
    print(f"  Numeric N    = {N_numeric_const:.3e} cycles")
    print("")
    print(f"Edge crack model:")
    print(f"  Numeric N    = {N_numeric_edge:.3e} cycles\n")

    print(f"Both sets of results logged to '{LOG_FILE}'.\n")

    # -------------
    # COMPARISON PLOT
    # -------------
    plt.figure()
    plt.plot(N_const_values, a_const_values * 1000.0, label="Constant Y model")
    plt.plot(N_edge_values, a_edge_values * 1000.0, label="Edge crack model")
    plt.xlabel("Number of Cycles, N")
    plt.ylabel("Crack Length, a (mm)")
    plt.title("Comparison of Fatigue Crack Growth Models\n"
              "Constant Y vs. Edge Crack Y(a/W)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================================================================
# MAIN MENU
# ================================================================

def main():
    """
    Main function implementing the text-based menu.
    """
    while True:
        print("\n==============================")
        print(" Paris Law Demonstration Menu")
        print("==============================")
        print("1) Simple model (constant geometry factor Y)")
        print("   - Analytic solution for N")
        print("   - Numerical integration")
        print("   - Logs run to CSV")
        print("")
        print("2) Edge-crack plate model (Y(a/W))")
        print("   - Numerical integration only")
        print("   - Logs run to CSV")
        print("")
        print("3) Compare BOTH models on the same plot")
        print("   - Uses same parameters where applicable")
        print("   - Logs BOTH to CSV")
        print("")
        print("4) Quit")
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        if choice == "1":
            run_simple_constant_Y_model()
        elif choice == "2":
            run_edge_crack_model()
        elif choice == "3":
            run_comparison_both_models()
        elif choice == "4":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

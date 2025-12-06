"""
goertler_number_calculator_with_plots.py

Educational script to calculate the *local Görtler number* and
to visualize how it changes with the radius of curvature (R)
and boundary-layer thickness (delta).

--------------------------------------------------------------
DEFINITION (one common form)
--------------------------------------------------------------
Local Görtler number:

    Go = Re_delta * sqrt( delta / R )

where

    Re_delta = (U_e * delta) / nu

and

    U_e   = local external / edge velocity of the boundary layer (m/s)
    delta = characteristic boundary-layer thickness (m)
    nu    = kinematic viscosity of the fluid (m^2/s)
    R     = radius of curvature of the wall (m, > 0 for this script)

--------------------------------------------------------------
SCRIPT FEATURES
--------------------------------------------------------------
Menu options:

  1) Enter U_e, delta, nu, R directly.
     -> Script calculates Re_delta and Go.

  2) Enter Re_delta, delta, R directly.
     -> Script calculates Go only.

  3) Plot Go vs R and Go vs delta.
     -> Students choose parameters and ranges; script shows:
          - Go as a function of R with delta held fixed.
          - Go as a function of delta with R held fixed.

  4) Quit.

The code is heavily commented for educational use.
"""

import math
import matplotlib.pyplot as plt  # For plotting Go vs R and Go vs delta


# ============================================================
# CORE CALCULATION FUNCTIONS
# ============================================================

def compute_re_delta(U_e, delta, nu):
    """
    Compute the local Reynolds number based on a boundary-layer thickness.

        Re_delta = (U_e * delta) / nu

    Parameters
    ----------
    U_e : float
        Local external (edge) velocity of the boundary layer. [m/s]
    delta : float
        Characteristic boundary-layer thickness. [m]
    nu : float
        Kinematic viscosity of the fluid. [m^2/s]

    Returns
    -------
    Re_delta : float
        Local Reynolds number based on delta (dimensionless).
    """
    return U_e * delta / nu


def compute_goertler_from_Re(Re_delta, delta, R):
    """
    Compute the local Görtler number from Re_delta, delta, and R:

        Go = Re_delta * sqrt( delta / R )

    Parameters
    ----------
    Re_delta : float
        Local Reynolds number based on delta (dimensionless).
    delta : float
        Boundary-layer thickness (m).
    R : float
        Radius of curvature of the wall (m).
        For concave walls, R is usually taken as positive.

    Returns
    -------
    Go : float
        Local Görtler number (dimensionless).
    """
    # Check R > 0 to avoid division by zero or negative inside sqrt
    if R <= 0.0:
        raise ValueError("Radius of curvature R must be > 0 for this formula.")

    sqrt_term = math.sqrt(delta / R)
    Go = Re_delta * sqrt_term
    return Go


def compute_goertler(U_e, delta, nu, R):
    """
    Convenience function to go from (U_e, delta, nu, R)
    directly to (Re_delta, Go).

    Steps:
        1) Re_delta = (U_e * delta) / nu
        2) Go       = Re_delta * sqrt(delta / R)

    Parameters
    ----------
    U_e : float
        Local external velocity (m/s).
    delta : float
        Boundary-layer thickness (m).
    nu : float
        Kinematic viscosity (m^2/s).
    R : float
        Radius of curvature of wall (m).

    Returns
    -------
    Re_delta : float
        Local Reynolds number based on delta (dimensionless).
    Go : float
        Görtler number (dimensionless).
    """
    Re_delta = compute_re_delta(U_e, delta, nu)
    Go = compute_goertler_from_Re(Re_delta, delta, R)
    return Re_delta, Go


# ============================================================
# INPUT HELPER
# ============================================================

def input_with_default(prompt, default, cast_type=float):
    """
    Prompt the user for input with a default value.

    If the user presses ENTER without typing anything, the
    default value is used. This is convenient for classroom
    demos where you want quick example values.

    Parameters
    ----------
    prompt : str
        Text shown to the user.
    default : any
        Default value if the user just presses ENTER.
    cast_type : callable
        Function used to convert the input string to the desired type,
        e.g. float, int, or str.

    Returns
    -------
    value : same type as 'default' (after casting).
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
# MENU OPTION 1: DIRECT INPUT OF U_e, delta, nu, R
# ============================================================

def option_1_direct_inputs():
    """
    Option 1:
      - Ask the user directly for U_e, delta, nu, R.
      - Compute Re_delta and Go.
      - Print results with basic interpretation.
    """
    print("\n=== OPTION 1: Direct Input of U_e, delta, nu, and R ===\n")
    print("All inputs are in SI units:")
    print("  U_e   in m/s      (edge velocity of boundary layer)")
    print("  delta in m        (boundary-layer thickness)")
    print("  nu    in m^2/s    (kinematic viscosity)")
    print("  R     in m        (radius of curvature of wall)")
    print("-------------------------------------------------------\n")

    # Typical example defaults
    U_e = input_with_default("Enter U_e (m/s)", 50.0, float)
    delta = input_with_default("Enter delta (m)", 0.01, float)   # 1 cm
    nu = input_with_default("Enter nu (m^2/s)", 1.5e-5, float)   # air ~ STP
    R = input_with_default("Enter R (m)", 1.0, float)            # 1 m radius

    # Basic safety checks
    if U_e <= 0.0:
        print("\nWARNING: U_e <= 0 is unphysical; using abs(U_e).\n")
        U_e = abs(U_e)

    if delta <= 0.0:
        print("\nERROR: delta must be > 0. Cannot compute.\n")
        return

    if nu <= 0.0:
        print("\nERROR: nu must be > 0. Cannot compute.\n")
        return

    if R <= 0.0:
        print("\nERROR: R must be > 0 for this formula. Cannot compute.\n")
        return

    # Compute
    Re_delta, Go = compute_goertler(U_e, delta, nu, R)

    # Present results
    print("\n--- RESULTS (Option 1) ---")
    print(f"U_e   = {U_e:.4e} m/s")
    print(f"delta = {delta:.4e} m")
    print(f"nu    = {nu:.4e} m^2/s")
    print(f"R     = {R:.4e} m")
    print("")
    print(f"Re_delta = (U_e * delta / nu) = {Re_delta:.4e}")
    print(f"Görtler number Go = Re_delta * sqrt(delta / R) = {Go:.4e}")
    print("--------------------------\n")

    print("Qualitative note (very rough):")
    print("  - Larger Go suggests a stronger tendency for centrifugal")
    print("    instabilities (Görtler vortices) in the boundary layer.")
    print("  - There is not a single universal 'critical' Go, but some")
    print("    references suggest instability can occur for Go on the")
    print("    order of ~0.3–1 and above, depending on the situation.\n")


# ============================================================
# MENU OPTION 2: INPUT Re_delta DIRECTLY
# ============================================================

def option_2_from_Re():
    """
    Option 2:
      - User already knows Re_delta (from another calculation).
      - User provides Re_delta, delta, and R.
      - Script computes Go:

            Go = Re_delta * sqrt(delta / R)
    """
    print("\n=== OPTION 2: Use Re_delta Directly ===\n")
    print("You will enter:")
    print("  Re_delta (dimensionless)")
    print("  delta (m)")
    print("  R (m)")
    print("The code will compute:")
    print("  Go = Re_delta * sqrt(delta / R)")
    print("----------------------------------------\n")

    Re_delta = input_with_default("Enter Re_delta (dimensionless)", 3.33e4, float)
    delta = input_with_default("Enter delta (m)", 0.01, float)
    R = input_with_default("Enter R (m)", 1.0, float)

    if delta <= 0.0:
        print("\nERROR: delta must be > 0. Cannot compute.\n")
        return

    if R <= 0.0:
        print("\nERROR: R must be > 0 for this formula. Cannot compute.\n")
        return

    if Re_delta <= 0.0:
        print("\nWARNING: Re_delta <= 0 is not physically meaningful.")
        print("Using abs(Re_delta).\n")
        Re_delta = abs(Re_delta)

    Go = compute_goertler_from_Re(Re_delta, delta, R)

    print("\n--- RESULTS (Option 2) ---")
    print(f"Re_delta = {Re_delta:.4e}")
    print(f"delta    = {delta:.4e} m")
    print(f"R        = {R:.4e} m")
    print("")
    print(f"Görtler number Go = Re_delta * sqrt(delta / R) = {Go:.4e}")
    print("--------------------------\n")

    print("Reminder: This is a *local* Görtler number based on the")
    print("local values of Re_delta, delta, and R.\n")


# ============================================================
# MENU OPTION 3: PARAMETRIC PLOTS (Go vs R, Go vs delta)
# ============================================================

def option_3_plots():
    """
    Option 3:
      - Perform a simple parametric study and create plots:
          1) Go vs R  (holding delta constant)
          2) Go vs delta (holding R constant)

      - For both plots, we fix:
          * U_e and nu (for computing Re_delta).
          * Then vary delta or R as specified.

      - This is excellent for students to visualize how curvature
        and boundary-layer thickness affect the Görtler number.
    """
    print("\n=== OPTION 3: Plot Go vs R and Go vs delta ===\n")
    print("We will use the formula:")
    print("    Go = (U_e * delta / nu) * sqrt(delta / R)")
    print("For the plots we need:")
    print("  - U_e (m/s)")
    print("  - nu (m^2/s)")
    print("  - A 'baseline' delta and R (for the fixed cases)")
    print("  - R range (for Go vs R plot)")
    print("  - delta range (for Go vs delta plot)\n")

    # Common fluid parameters
    U_e = input_with_default("Enter U_e (m/s)", 50.0, float)
    nu = input_with_default("Enter nu (m^2/s)", 1.5e-5, float)

    # Baseline values for holding constant
    delta_fixed = input_with_default("Enter baseline delta for Go vs R (m)", 0.01, float)
    R_fixed = input_with_default("Enter baseline R for Go vs delta (m)", 1.0, float)

    # Range for R (Go vs R)
    R_min = input_with_default("Enter R_min for Go vs R (m)", 0.2, float)
    R_max = input_with_default("Enter R_max for Go vs R (m)", 5.0, float)

    # Range for delta (Go vs delta)
    delta_min = input_with_default("Enter delta_min for Go vs delta (m)", 0.001, float)
    delta_max = input_with_default("Enter delta_max for Go vs delta (m)", 0.05, float)

    # Number of points for each curve
    n_points_R = int(input_with_default("Number of points in Go vs R plot", 100, int))
    n_points_delta = int(input_with_default("Number of points in Go vs delta plot", 100, int))

    # --- Basic validation ---
    if U_e <= 0.0:
        print("\nWARNING: U_e <= 0 is unphysical; using abs(U_e).\n")
        U_e = abs(U_e)

    if nu <= 0.0:
        print("\nERROR: nu must be > 0. Cannot generate plots.\n")
        return

    if delta_fixed <= 0.0:
        print("\nERROR: baseline delta for Go vs R must be > 0.\n")
        return

    if R_fixed <= 0.0:
        print("\nERROR: baseline R for Go vs delta must be > 0.\n")
        return

    if R_min <= 0.0 or R_max <= 0.0 or R_max <= R_min:
        print("\nERROR: invalid R range for Go vs R plot.\n")
        return

    if delta_min <= 0.0 or delta_max <= 0.0 or delta_max <= delta_min:
        print("\nERROR: invalid delta range for Go vs delta plot.\n")
        return

    # 1) Go vs R (holding delta = delta_fixed constant)
    R_values = [R_min + (R_max - R_min) * i / (n_points_R - 1) for i in range(n_points_R)]
    Go_vs_R = []

    for R in R_values:
        # For each R, compute Re_delta and Go with delta_fixed
        Re_delta = compute_re_delta(U_e, delta_fixed, nu)
        Go = compute_goertler_from_Re(Re_delta, delta_fixed, R)
        Go_vs_R.append(Go)

    # 2) Go vs delta (holding R = R_fixed constant)
    delta_values = [delta_min + (delta_max - delta_min) * i / (n_points_delta - 1)
                    for i in range(n_points_delta)]
    Go_vs_delta = []

    for delta in delta_values:
        Re_delta = compute_re_delta(U_e, delta, nu)
        Go = compute_goertler_from_Re(Re_delta, delta, R_fixed)
        Go_vs_delta.append(Go)

    # --- Plot 1: Go vs R ---
    plt.figure()
    plt.plot(R_values, Go_vs_R)
    plt.xlabel("Radius of Curvature R (m)")
    plt.ylabel("Görtler Number Go (dimensionless)")
    plt.title(f"Go vs R (delta fixed = {delta_fixed} m, U_e = {U_e} m/s, nu = {nu} m^2/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Go vs delta ---
    plt.figure()
    plt.plot(delta_values, Go_vs_delta)
    plt.xlabel("Boundary-layer thickness delta (m)")
    plt.ylabel("Görtler Number Go (dimensionless)")
    plt.title(f"Go vs delta (R fixed = {R_fixed} m, U_e = {U_e} m/s, nu = {nu} m^2/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nTwo figures were generated:")
    print("  1) Go vs R (with delta held fixed)")
    print("  2) Go vs delta (with R held fixed)\n")
    print("Students can observe:")
    print("  - How decreasing R (tighter curvature) generally increases Go.")
    print("  - How increasing delta (thicker boundary layer) also increases Go.")
    print("This connects the math definition of Go with physical intuition.\n")


# ============================================================
# MAIN MENU / PROGRAM ENTRY POINT
# ============================================================

def main():
    """
    MAIN FUNCTION

    Implements a simple text-based menu so that students can
    choose how they want to work with the Görtler number:

      1) Compute from U_e, delta, nu, R.
      2) Compute from Re_delta, delta, R.
      3) Visualize Go vs R and Go vs delta.
      4) Quit.
    """
    while True:
        print("==============================================")
        print("          Görtler Number Calculator           ")
        print("==============================================")
        print("Please choose an option:")
        print("  1) Enter U_e, delta, nu, R directly")
        print("     -> Script calculates Re_delta and Go")
        print("")
        print("  2) Enter Re_delta, delta, R directly")
        print("     -> Script calculates Go only")
        print("")
        print("  3) Plot Go vs R and Go vs delta")
        print("     -> Shows how Go changes with curvature and thickness")
        print("")
        print("  4) Quit")
        choice = input("Your choice (1, 2, 3, or 4): ").strip()

        if choice == "1":
            option_1_direct_inputs()
        elif choice == "2":
            option_2_from_Re()
        elif choice == "3":
            option_3_plots()
        elif choice == "4":
            print("\nExiting Görtler number calculator. Goodbye!\n")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.\n")


# Only run the menu if the script is executed directly,
# not if it is imported as a module.
if __name__ == "__main__":
    main()

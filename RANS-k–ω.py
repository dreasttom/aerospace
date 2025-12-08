"""
Simple RANS k–ω Turbulence Model Calculator with GUI

This script is meant as a TEACHING AID for students.

It demonstrates the basic k–ω turbulence model relationship:

    Turbulent (eddy) viscosity:
        ν_t = k / ω

    where:
        k     = turbulent kinetic energy       [m^2/s^2]
        ω     = specific dissipation rate      [1/s]
        ν_t   = turbulent (eddy) viscosity     [m^2/s]

Optionally, the student can also enter a laminar (molecular) kinematic viscosity ν
so that the code computes:

        ν_eff = ν + ν_t

This is *not* a full CFD solver (no Navier–Stokes), just a small closure-model
calculator with a GUI.

Dependencies:
    - Tkinter (usually included with Python; if not, install e.g. python3-tk)
"""

import sys
import traceback

# --- Safe imports for Tkinter ------------------------------------------------
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("ERROR: Tkinter is not available on this system.")
    print("On many Linux systems, you may need to install it, e.g.:")
    print("    sudo apt-get install python3-tk")
    sys.exit(1)


# =============================================================================
#  Core physics function: RANS k–ω model
# =============================================================================

def nu_t_k_omega(k, omega):
    """
    Compute turbulent viscosity ν_t for the RANS k–ω model.

    Formula:
        ν_t = k / ω

    Parameters
    ----------
    k : float
        Turbulent kinetic energy [m^2/s^2].
    omega : float
        Specific dissipation rate [1/s].

    Returns
    -------
    nu_t : float
        Turbulent (eddy) viscosity [m^2/s].

    Raises
    ------
    ValueError
        If k <= 0 or omega <= 0 (unphysical for this simple model).
    """
    # --- Basic physical checks ----------------------------------------------
    if k <= 0.0:
        raise ValueError("Turbulent kinetic energy k must be positive (k > 0).")
    if omega <= 0.0:
        raise ValueError("Specific dissipation rate ω must be positive (ω > 0).")

    # Simple k–ω formula
    return k / omega


# =============================================================================
#  GUI class
# =============================================================================

class KOmegaGUI:
    """
    Tkinter-based GUI for the RANS k–ω turbulence model.

    The GUI lets students:
        - Enter k and ω
        - Optionally enter laminar kinematic viscosity ν
        - Click a button to compute ν_t and (optionally) ν_eff
        - See results displayed in the window
    """

    def __init__(self, master):
        """
        Set up the GUI elements (labels, text boxes, button, etc.).
        """
        self.master = master
        master.title("RANS k–ω Turbulence Model Calculator")

        # ------------------------------
        # Row 0: Title / description
        # ------------------------------
        title_label = tk.Label(
            master,
            text="RANS k–ω Turbulence Model: ν_t = k / ω",
            font=("Arial", 12, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 5))

        # Small explanatory note
        note_label = tk.Label(
            master,
            text="Enter k (m²/s²) and ω (1/s). Optionally enter laminar ν (m²/s).",
            fg="gray"
        )
        note_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # ------------------------------
        # Input fields for k, ω, ν
        # ------------------------------

        # Turbulent kinetic energy k
        self.k_var = tk.StringVar(value="1.0")
        self._add_labeled_entry(
            label_text="k  (turb. kinetic energy, m²/s²):",
            textvariable=self.k_var,
            row=2
        )

        # Specific dissipation rate ω
        self.omega_var = tk.StringVar(value="1.0")
        self._add_labeled_entry(
            label_text="ω  (specific dissipation rate, 1/s):",
            textvariable=self.omega_var,
            row=3
        )

        # Optional laminar viscosity ν
        # If blank, we will skip effective viscosity calculation.
        self.nu_lam_var = tk.StringVar(value="")
        self._add_labeled_entry(
            label_text="ν  (laminar kinematic viscosity, m²/s, optional):",
            textvariable=self.nu_lam_var,
            row=4
        )

        # ------------------------------
        # Button to run the calculation
        # ------------------------------
        run_button = tk.Button(
            master,
            text="Run k–ω Calculation",
            command=self.run_calculation
        )
        run_button.grid(row=5, column=0, columnspan=2, pady=10)

        # ------------------------------
        # Labels for results
        # ------------------------------
        self.result_label_nut = tk.Label(
            master,
            text="ν_t (eddy viscosity) = ?",
            fg="blue"
        )
        self.result_label_nut.grid(row=6, column=0, columnspan=2, pady=(5, 2))

        self.result_label_nueff = tk.Label(
            master,
            text="ν_eff (laminar + turbulent) = ?",
            fg="blue"
        )
        self.result_label_nueff.grid(row=7, column=0, columnspan=2, pady=(2, 10))

        # ------------------------------
        # Final note to students
        # ------------------------------
        final_note = tk.Label(
            master,
            text=(
                "Reminder: This is just the closure relation for ν_t.\n"
                "A full RANS solver would also solve PDEs for k and ω."
            ),
            fg="gray"
        )
        final_note.grid(row=8, column=0, columnspan=2, pady=(0, 10))

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    def _add_labeled_entry(self, label_text, textvariable, row):
        """
        Create a label and entry widget side-by-side.

        Parameters
        ----------
        label_text : str
            Text for the label.
        textvariable : tk.StringVar
            Tkinter StringVar bound to the entry.
        row : int
            Row index in the grid layout.
        """
        label = tk.Label(self.master, text=label_text)
        label.grid(row=row, column=0, sticky="e", padx=5, pady=3)

        entry = tk.Entry(self.master, textvariable=textvariable, width=25)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=3)

    def _parse_float(self, string_var, name, allow_empty=False):
        """
        Safely parse a float from a tk.StringVar.

        Parameters
        ----------
        string_var : tk.StringVar
            The variable holding the text.
        name : str
            Name of the field, used in error messages.
        allow_empty : bool
            If True, allow an empty string and return None.

        Returns
        -------
        float or None
            Parsed float, or None if allow_empty=True and the field is empty.

        Raises
        ------
        ValueError
            If parsing fails (and allow_empty=False) or the value is not a valid float.
        """
        text = string_var.get().strip()

        if allow_empty and text == "":
            # Caller wants to allow an empty field (e.g., optional input).
            return None

        try:
            value = float(text)
        except ValueError:
            # Show a GUI message and raise again so we can handle it.
            messagebox.showerror(
                "Invalid Input",
                f"{name} must be a number. Got: {text or '<empty>'}"
            )
            raise

        return value

    # ---------------------------------------------------------------------
    # Core GUI callback
    # ---------------------------------------------------------------------

    def run_calculation(self):
        """
        This method is called when the user presses the
        "Run k–ω Calculation" button.

        It:
            1) Reads and validates inputs from the GUI.
            2) Computes ν_t using the k–ω formula.
            3) Optionally computes ν_eff = ν + ν_t if ν is provided.
            4) Updates labels with results.
            5) Uses try/except + messagebox for error handling.
        """
        try:
            # 1) Read and parse k and ω
            k = self._parse_float(self.k_var, "k")
            omega = self._parse_float(self.omega_var, "ω")

            # 2) Compute turbulent viscosity ν_t
            nu_t = nu_t_k_omega(k, omega)

            # 3) Optionally read laminar viscosity ν
            nu_lam = self._parse_float(self.nu_lam_var, "ν (laminar viscosity)", allow_empty=True)

            # Update ν_t label first
            self.result_label_nut.config(
                text=f"ν_t (eddy viscosity) = {nu_t:.4e} m²/s"
            )

            # If laminar ν is provided, compute ν_eff
            if nu_lam is not None:
                if nu_lam < 0.0:
                    raise ValueError("Laminar viscosity ν must be non-negative.")
                nu_eff = nu_lam + nu_t
                self.result_label_nueff.config(
                    text=f"ν_eff (laminar + turbulent) = {nu_eff:.4e} m²/s"
                )
            else:
                # If no laminar ν given, say "N/A"
                self.result_label_nueff.config(
                    text="ν_eff (laminar + turbulent) = N/A (no ν provided)"
                )

        except Exception as e:
            # Print the traceback to the console (useful if you run from terminal)
            traceback.print_exc()
            # Show a message box to the user
            messagebox.showerror(
                "Error during calculation",
                f"An error occurred:\n{str(e)}"
            )


# =============================================================================
#  Main program entry point
# =============================================================================

def main():
    """
    Entry point: create the Tkinter root window and start the GUI loop.
    """
    root = tk.Tk()
    app = KOmegaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

import sympy as sp
import numpy as np
from pathlib import Path
from biosym.utils.rnea import get_rnea_equations
from biosym.utils.useful_functions import read_mot
from scipy.signal import butter, filtfilt
import jax
import jax.numpy as jnp
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BaseParamCompiler:
    """Manage substitution, simplification, caching, and compilation with base parameters."""

    def __init__(
        self,
        model_path: str | None = None,
        model=None,
        cache_dir: str = "data/cache/",
        shoe_weight: float = 0.0,
    ):
        """Initialize compiler with a model instance or model path and optional cache directory."""
        self.model_path = model_path
        self.shoe_weight = float(shoe_weight or 0.0)
        if model is None:
            if model_path is None:
                raise ValueError("Provide model_path or model instance.")
            from biosym.model.model import load_model

            model = load_model(self.model_path, force_rebuild=True)

        self.model = model
        self.cache_dir = self._project_path(cache_dir)
        
        # State populated by build()
        self.substituted_eqs = None
        self.substituted_eqs_lmb = None
        self.substituted_eqs_external = None
        self.substituted_eqs_zeroed = None
        base_param_symbols = None
        if getattr(model, "base_params", {}).get("n", 0) > 0:
            base_param_symbols = [
                model._v[i]
                for i in range(
                    model.base_params["idx"],
                    model.base_params["idx"] + model.base_params["n"],
                )
            ]
        self.mapping_dict, self.base_params = inertial_to_base_2d_mapping(
            model=self.model,
            base_param_symbols=base_param_symbols,
            shoe_weight=self.shoe_weight,
        )
        self.q_syms = model._v[model.coordinates["idx"] : model.coordinates["idx"] + model.coordinates["n"]]
        self.dq_syms = model._v[model.speeds["idx"] : model.speeds["idx"] + model.speeds["n"]]
        self.ddq_syms = model._v[model.accs["idx"] : model.accs["idx"] + model.accs["n"]]
        self.g_syms = model._v[model.g["idx"] : model.g["idx"] + model.g["n"]]
        self.ext_loads_syms = model._v[model.ext_forces["idx"] : model.ext_forces["idx"] + model.ext_forces["n"]] + model._v[model.ext_torques["idx"] : model.ext_torques["idx"] + model.ext_torques["n"]]
        offset_meta = getattr(model, "offset", None)
        if offset_meta and offset_meta.get("n", 0) > 0:
            self.offset_syms = model._v[offset_meta["idx"] : offset_meta["idx"] + offset_meta["n"]]
            self.offset_defaults = [
                float(val)
                for val in model.default_values[offset_meta["idx"] : offset_meta["idx"] + offset_meta["n"]]
            ]
        else:
            self.offset_syms = []
            self.offset_defaults = []
        self.all_input_symbols = self.q_syms + self.dq_syms + self.ddq_syms + self.base_params

    @staticmethod
    def _is_trial_collection(value) -> bool:
        """Return True for a list/tuple of trial inputs, not for a single path/DataFrame."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _validate_matching_trial_collections(ik, grf) -> None:
        if not BaseParamCompiler._is_trial_collection(ik) or not BaseParamCompiler._is_trial_collection(grf):
            raise ValueError("For multi-trial external loads, provide both ik and grf as lists/tuples.")
        if len(ik) != len(grf):
            raise ValueError(
                f"IK and GRF trial collections must have the same length, got {len(ik)} and {len(grf)}."
            )
        if len(ik) == 0:
            raise ValueError("Trial collections must contain at least one trial.")

    def _external_load_subs(self) -> dict:
        """Return a substitution map that zeros external forces and moments."""
        return {sym: sp.Integer(0) for sym in self.ext_loads_syms}

    def _offset_subs(self) -> dict:
        """Return a substitution map for constant segment offsets."""
        if not self.offset_syms:
            return {}
        return {
            sym: sp.Float(val)
            for sym, val in zip(self.offset_syms, self.offset_defaults)
        }

    def _project_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        parts = path.parts
        if "data" in parts:
            data_idx = parts.index("data")
            if len(parts) > data_idx + 1 and parts[data_idx + 1] == "cache":
                return PROJECT_ROOT / Path(*parts[data_idx:])
        return PROJECT_ROOT / path

    def _cache_file(self, filepath: str | Path | None = None, default_name: str = "kanes_base.cpkl") -> Path:
        if filepath is None:
            path = self.cache_dir
            filepath = path if path.suffix == ".cpkl" else path / default_name
        return self._project_path(filepath)

    def load_expressions(self, filepath: str | Path | None = None):
        """Load cached base-parameterized expressions.

        Defaults to the Kane's-method cache file, ``kanes_base.cpkl``.
        """
        filepath = self._cache_file(filepath, "kanes_base.cpkl")
        if not filepath.exists():
            raise FileNotFoundError(f"CPKL file not found: {filepath}")
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        cached_base_param_names = payload.get("base_param_names")
        current_base_param_names = [str(sym) for sym in self.base_params]
        if cached_base_param_names is not None and cached_base_param_names != current_base_param_names:
            raise ValueError(
                "Cached base-parameter expressions do not match this model's base-parameter symbols."
            )
        self.substituted_eqs = payload.get("substituted_eqs", payload.get("expressions"))
        self.base_params = payload.get("base_params", self.base_params)
        return self.substituted_eqs

    def export_expressions(
        self,
        expressions,
        filepath: str | Path | None = None,
    ) -> None:
        """Save base-parameterized SymPy expressions and metadata.

        Defaults to the Kane's-method cache file, ``kanes_base.cpkl``.
        """
        filepath = self._cache_file(filepath, "kanes_base.cpkl")
        payload = {
            "expressions": expressions,
            "substituted_eqs": expressions,
            "base_params": self.base_params,
            "base_param_names": [str(sym) for sym in self.base_params],
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def _substitute_base_parameters(
        self,
        expressions,
    ):
        """Apply the 2D base-parameter mapping to dict/list/Matrix/scalar expressions."""

        def substitute_one(expr):
            substituted = expr.subs(self.mapping_dict)
            substituted = substituted.expand()
            substituted = sp.simplify(substituted)
            return substituted

        if isinstance(expressions, dict):
            return {name: substitute_one(expr) for name, expr in expressions.items()}
        if isinstance(expressions, sp.MatrixBase):
            return expressions.applyfunc(substitute_one)
        if isinstance(expressions, (list, tuple)):
            return type(expressions)(substitute_one(expr) for expr in expressions)

        return substitute_one(expressions)

    def build(
        self,
        expressions,
        force_rebuild: bool = False,
    ):
        """Load or build cached Kane's-method base-parameterized expressions."""
        filepath = self._cache_file(None, "kanes_base.cpkl")
        if not force_rebuild and filepath.exists():
            print(f"Loading base-parameterized expressions from: {filepath.resolve()}")
            self.substituted_eqs = self.load_expressions()
            return self.substituted_eqs

        print(f"Building base-parameterized expressions and saving to: {filepath.resolve()}")
        self.substituted_eqs = self._substitute_base_parameters(
            expressions,
        )

        self.export_expressions(self.substituted_eqs)

        return self.substituted_eqs

    def _ik_to_jnp(
        self,
        df,
        mean_suffix: str = "_mean",
        time_column: str | None = None,
        lowpass_hz: float | None = None,
        sample_rate_hz: float | None = 100.0,
        dt: float | None = None,
    ) -> jnp.ndarray:
        """Build a [q, dq, ddq] trajectory from a DataFrame or .mot file path.

        Optionally applies a low-pass filter to q before differentiation.
        """

        def _zero_phase_butter_lowpass(
            data: np.ndarray,
            cutoff_hz: float,
            sample_rate: float,
        ) -> np.ndarray:
        
            if cutoff_hz <= 0:
                return data
            nyquist = 0.5 * sample_rate
            if cutoff_hz >= nyquist:
                raise ValueError(
                    f"lowpass_hz must be < Nyquist ({nyquist:.3g} Hz), got {cutoff_hz}."
                )

            wn = cutoff_hz / nyquist
            b, a = butter(N=4, Wn=wn, btype="low")
            padlen = min(data.shape[0] - 1, 3 * (max(len(a), len(b)) - 1))
            if padlen < 1:
                return data
            return filtfilt(b, a, data, axis=0, padlen=padlen)


        if isinstance(df, (str, Path)):
            df = read_mot(str(df))

        # Model coordinates are stored as q_<name>; motion files usually use <name>.
        coord_aliases = {
            "hip_r": "hip_flexion_r",
            "knee_r": "knee_angle_r",
            "ankle_r": "ankle_angle_r",
            "hip_l": "hip_flexion_l",
            "knee_l": "knee_angle_l",
            "ankle_l": "ankle_angle_l",
        }

        q_names = [name.removeprefix("q_") for name in self.model.coordinates["names"]]
        q_cols = []
        missing = []
        for name in q_names:
            base_name = coord_aliases.get(name, name)
            mean_col = f"{base_name}{mean_suffix}"
            if mean_col in df.columns:
                q_cols.append(mean_col)
            elif base_name in df.columns:
                q_cols.append(base_name)
            else:
                missing.append(mean_col)

        if missing:
            raise ValueError(f"Missing kinematics columns: {missing}")

        q = np.asarray(df[q_cols].to_numpy(), dtype=float)

        if time_column is None:
            if "time" in df.columns:
                time_column = "time"
            elif df.index.name == "time":
                time_column = "__index__"

        if time_column == "__index__":
            time_np = np.asarray(df.index.to_numpy(), dtype=float)
        elif time_column is not None:
            time_np = np.asarray(df[time_column].to_numpy(), dtype=float)
        else:
            time_np = None

        # Low-pass filter before differentiation to reduce noise amplification.
        if lowpass_hz is not None:
            if time_np is not None and time_np.size > 1:
                sample_rate = 1.0 / float(np.median(np.diff(time_np)))
            else:
                if dt is None:
                    if sample_rate_hz is None:
                        sample_rate_hz = 100.0
                    dt = 1.0 / float(sample_rate_hz)
                sample_rate = 1.0 / float(dt)
            if sample_rate <= 0:
                raise ValueError(f"Invalid sample rate for filtering: {sample_rate}")
            q = _zero_phase_butter_lowpass(q, lowpass_hz, sample_rate)

        q = jnp.asarray(q)

        if time_np is not None:
            time = jnp.asarray(time_np)
        else:
            time = None

        if time is not None:
            dq = jnp.gradient(q, time, axis=0)
            ddq = jnp.gradient(dq, time, axis=0)
        else:
            if dt is None:
                if sample_rate_hz is None:
                    sample_rate_hz = 100.0
                dt = 1.0 / float(sample_rate_hz)
            dq = jnp.gradient(q, dt, axis=0)
            ddq = jnp.gradient(dq, dt, axis=0)

        return jnp.concatenate([q, dq, ddq], axis=1)

    def _grf_to_jnp(
        self,
        df,
        left_force_prefix: str = "1_ground_force",
        right_force_prefix: str = "ground_force",
        left_torque_prefix: str = "1_ground_torque",
        right_torque_prefix: str = "ground_torque",
    ) -> jnp.ndarray:

        """Build an external-load trajectory from a GRF .mot file or DataFrame.

        Columns are ordered to match model external loads: forces (x,y,z per body)
        followed by torques (x,y,z per body). By default, the method assumes:
        - right leg columns are prefixed with 'ground_force'/'ground_torque'
        - left leg columns are prefixed with '1_ground_force'/'1_ground_torque'
        """
        if isinstance(df, (str, Path)):
            df = read_mot(str(df))[1:]

        force_cols: list[str] = []
        torque_cols: list[str] = []
        missing: list[str] = []

        def _resolve_col(base: str) -> str | None:
            if base in df.columns:
                return base
            mean_col = f"{base}_mean"
            if mean_col in df.columns:
                return mean_col
            return None

        body_names = []
        for idx, force_name in enumerate(self.model.ext_forces["names"]):
            if idx % 3 != 0:
                continue
            body_name = "_".join(force_name.split("_")[1:-1])
            body_names.append(body_name)

        if len(body_names) != 2:
            raise ValueError(
                "Hard-coded left/right GRF mapping expects exactly 2 external force bodies, "
                f"got {len(body_names)}."
            )

        side_prefixes = [
            (left_force_prefix, left_torque_prefix),
            (right_force_prefix, right_torque_prefix),
        ]

        for force_prefix, torque_prefix in side_prefixes:

            for dim in ("x", "y", "z"):
                force_base = f"{force_prefix}_v{dim}"
                resolved = _resolve_col(force_base)
                if resolved is not None:
                    force_cols.append(resolved)
                else:
                    missing.append(force_base)

            for dim in ("x", "y", "z"):
                torque_base = f"{torque_prefix}_{dim}"
                resolved = _resolve_col(torque_base)
                if resolved is not None:
                    torque_cols.append(resolved)
                else:
                    missing.append(torque_base)

        if missing:
            raise ValueError(f"Missing GRF columns: {sorted(set(missing))}")

        return jnp.asarray(df[force_cols + torque_cols].to_numpy())

    def rnea_base_equations(
        self,
        cpkl_path: str | Path | None = None,
    ):
        """Load or build cached base-parameterized RNEA equations and compile them."""
        cache_file = self._cache_file(cpkl_path, "rnea_base.cpkl")
        if cache_file.exists():
            try:
                print(f"Loading base-parameterized RNEA expressions from: {cache_file.resolve()}")
                self.substituted_eqs = self.load_expressions(cache_file)
            except Exception as exc:
                print(
                    "Could not load base-parameterized RNEA expressions from "
                    f"{cache_file.resolve()}: {type(exc).__name__}: {exc}"
                )
                self.substituted_eqs = None

        if self.substituted_eqs is None:
            print(f"Building base-parameterized RNEA expressions and saving to: {cache_file.resolve()}")
            self.substituted_eqs = self._substitute_base_parameters(
                get_rnea_equations(self.model),
            )
            self.export_expressions(self.substituted_eqs, cache_file)

        self.substituted_exteqs_ = {}
        self.substituted_kineqs = {}
        self.substituted_kineqs_lmb = {}
        self.substituted_exteqs_lmb = {}
        self.substituted_eqs_lmb = {}

        ext_syms = list(self.ext_loads_syms)
        all_symbols = list(self.all_input_symbols) + list(self.g_syms) + ext_syms + list(self.offset_syms)

        for joint_name, eq in (self.substituted_eqs or {}).items():
            eq_expanded = sp.expand(eq)
            if ext_syms:
                ext_terms = [
                    term
                    for term in eq_expanded.as_ordered_terms()
                    if term.has(*ext_syms)
                ]
                external_only_eq = sp.Add(*ext_terms) if ext_terms else sp.Integer(0)
            else:
                external_only_eq = sp.Integer(0)

            kinematic_only_eq = (eq_expanded - external_only_eq).subs(self._offset_subs())

            self.substituted_eqs[joint_name] = eq
            self.substituted_exteqs_[joint_name] = external_only_eq
            self.substituted_kineqs[joint_name] = kinematic_only_eq
            lam_fn = sp.lambdify(all_symbols, eq, modules="jax", cse=True)

            kin_symbols = list(self.all_input_symbols) + list(self.g_syms)
            lam_fn_kin = sp.lambdify(kin_symbols, kinematic_only_eq, modules="jax", cse=True)
            ext_symbols = [sym for sym in all_symbols if sym in external_only_eq.free_symbols]
            lam_fn_ext = sp.lambdify(ext_symbols, external_only_eq, modules="jax", cse=True)
            
            self.substituted_eqs_lmb[joint_name] = lam_fn
            self.substituted_kineqs_lmb[joint_name] = lam_fn_kin
            self.substituted_exteqs_lmb[joint_name] = lam_fn_ext

        return self.substituted_eqs_lmb, self.substituted_kineqs_lmb, self.substituted_exteqs_lmb

    def calculate_regressor_matrix(
        self,
        ik: str | Path | object,
        mean_suffix: str = "_mean",
        time_column: str | None = None,
        lowpass_hz: float | None = None,
        sample_rate_hz: float | None = 100.0,
        dt: float | None = None,
        frame_stride: int | None = None,
        cpkl_path: str | Path | None = None,
    ):
        """
        Calculate the stacked regressor matrix for a trajectory of kinematics using vmap.
        
        Parameters
        ----------
        rnea_equations: Loaded RNEA equations with base parameter substitutions and jitted lambdified functions.
        kinematics_trajectory : jnp.ndarray | pandas.DataFrame | str | pathlib.Path
            A 2D array of shape (T, num_kinematics) where T is the number of time steps.
            num_kinematics is the total number of [q, dq, ddq] variables.
            If a DataFrame is provided, it must include mean kinematics columns
            (e.g., pelvis_tx_mean) for each coordinate. Velocities and accelerations
            are computed with finite differences using `time_column` or `dt`.
            If a path is provided, it is loaded with `read_mot`.
        mean_suffix : str, default "_mean"
            Suffix used to locate mean kinematics columns in a DataFrame input.
        time_column : str | None, default None
            Column name to use as time for finite differences. If None and a DataFrame
            is provided, "time" is used when present, otherwise the index is used when
            it is named "time".
        lowpass_hz : float | None, default 6.0
            Low-pass cutoff (Hz) applied to q before differentiation. Use None to disable.
        sample_rate_hz : float | None, default 100.0
            Sampling rate used when no time column is available.
        dt : float | None, default None
            Time step to use when no time column is available in the DataFrame.
        frame_stride : int | None, default None
            Optional frame stride applied to the final regressor matrix. This happens
            after filtering and differentiation. If None, it is computed from
            `lowpass_hz` and the sampling rate. Example: 10 keeps every 10th row.
        cpkl_path : str | pathlib.Path | None, default None
            Optional path to a cached base-parameterized RNEA file. If not provided,
            `self.cache_dir` is used to locate the cache.
            
        Returns
        -------
        Y_stacked : jnp.ndarray
            The vertically stacked regressor matrix of shape (T * 3, num_base_params).
        """

        if self._is_trial_collection(ik):
            if len(ik) == 0:
                raise ValueError("IK trial collection must contain at least one trial.")
            trial_matrices = [
                self.calculate_regressor_matrix(
                    trial_ik,
                    mean_suffix=mean_suffix,
                    time_column=time_column,
                    lowpass_hz=lowpass_hz,
                    sample_rate_hz=sample_rate_hz,
                    dt=dt,
                    frame_stride=frame_stride,
                    cpkl_path=cpkl_path,
                )
                for trial_ik in ik
            ]
            return jnp.concatenate(trial_matrices, axis=0)
        
        self.rnea_base_equations(cpkl_path)

        # Build kinematics array from a .mot path or DataFrame
        if isinstance(ik, (str, Path)) or hasattr(ik, "columns"):
            kinematics_trajectory = self._ik_to_jnp(
                ik,
                mean_suffix=mean_suffix,
                time_column=time_column,
                lowpass_hz=lowpass_hz,
                sample_rate_hz=sample_rate_hz,
                dt=dt,
            )
        else:
            raise ValueError("Inverse Kinematics input must be a .mot path or DataFrame")
        
        pelvis_tx_fn = self.substituted_kineqs_lmb.get("pelvis_tx")
        pelvis_ty_fn = self.substituted_kineqs_lmb.get("pelvis_ty")
        pelvis_tilt_fn = self.substituted_kineqs_lmb.get("pelvis_tilt")
        
        if not all([pelvis_tx_fn, pelvis_ty_fn, pelvis_tilt_fn]):
            raise ValueError("Missing pelvis torque equations")

        g_values = jnp.asarray(self.model.default_values[self.model.g["idx"] : self.model.g["idx"] + self.model.g["n"]])

        # Base function for a SINGLE time step
        def pelvis_torques_fn(kinematics, base_params):
            flat_inputs = jnp.concatenate([kinematics, base_params, g_values])
            tx = pelvis_tx_fn(*flat_inputs)
            ty = pelvis_ty_fn(*flat_inputs)
            tilt = pelvis_tilt_fn(*flat_inputs)
            return jnp.array([tx, ty, tilt])

        # Compute the Jacobian for a single time step (w.r.t base parameters)
        jacobian_fn = jax.jacfwd(pelvis_torques_fn, argnums=1)
        
        # map over the 0th axis (time) of the first argument (kinematics_trajectory)
        vmap_jacobian_fn = jax.jit(jax.vmap(jacobian_fn, in_axes=(0, None)))
        
        # evaluate with dummy base parameters (e.g., zeros) to get the regressor matrix structure
        num_base_params = len(self.base_params)
        dummy_base_params = jnp.zeros(num_base_params)
        
        # Y_trajectory shape: (T, 3, num_base_params)
        Y_trajectory = vmap_jacobian_fn(kinematics_trajectory, dummy_base_params)

        if frame_stride is None:
            if lowpass_hz is not None and lowpass_hz > 0:
                if sample_rate_hz is None:
                    if dt is None:
                        sample_rate_hz = 100.0
                    else:
                        sample_rate_hz = 1.0 / float(dt)
                target_rate = 2.5 * float(lowpass_hz)
                frame_stride = int(np.ceil(float(sample_rate_hz) / target_rate))
        if frame_stride is not None:
            if frame_stride < 1:
                raise ValueError("frame_stride must be >= 1")
            Y_trajectory = Y_trajectory[::frame_stride, :, :]

        # stack vertically
        Y_stacked = Y_trajectory.reshape(-1, num_base_params)

        return Y_stacked

    def calculate_external_loads_matrix(
        self,
        ik: str | Path | object,
        grf: str | Path | object,
        mean_suffix: str = "_mean",
        time_column: str | None = None,
        lowpass_hz: float | None = None,
        sample_rate_hz: float | None = 100.0,
        dt: float | None = None,
        frame_stride: int | None = None,
        cpkl_path: str | Path | None = None,
    ) -> jnp.ndarray:
        """
        Calculate stacked external-load contribution (J_ext * f_ext) for pelvis equations.

        Parameters
        ----------
        ik : str | pathlib.Path | pandas.DataFrame
            Inverse kinematics data as a .mot path or a DataFrame.
        grf : str | pathlib.Path | pandas.DataFrame
            Ground reaction force data as a .mot path or a DataFrame.
        lowpass_hz : float | None, default 6.0
            Low-pass cutoff (Hz) applied to q before differentiation. Use None to disable.
        sample_rate_hz : float | None, default 100.0
            Sampling rate used when no time column is available.
        dt : float | None, default None
            Time step to use when no time column is available in the DataFrame.
        frame_stride : int | None, default None
            Optional frame stride applied to the final torque vector. This happens
            after filtering and differentiation. If None, it is computed from
            `lowpass_hz` and the sampling rate. Example: 10 keeps every 10th row.
        cpkl_path : see calculate_regressor_matrix
        base_params : jnp.ndarray | None
            Optional base-parameter values; defaults to zeros.

        Returns
        -------
        tau_ext_stacked : jnp.ndarray
            Stacked external-load contribution with shape (T * 3,).
        """

        if self._is_trial_collection(ik) or self._is_trial_collection(grf):
            self._validate_matching_trial_collections(ik, grf)
            trial_vectors = [
                self.calculate_external_loads_matrix(
                    trial_ik,
                    trial_grf,
                    mean_suffix=mean_suffix,
                    time_column=time_column,
                    lowpass_hz=lowpass_hz,
                    sample_rate_hz=sample_rate_hz,
                    dt=dt,
                    frame_stride=frame_stride,
                    cpkl_path=cpkl_path,
                )
                for trial_ik, trial_grf in zip(ik, grf)
            ]
            return jnp.concatenate(trial_vectors, axis=0)

        self.rnea_base_equations(cpkl_path)

        # Build kinematics array from a .mot path or DataFrame
        if isinstance(ik, (str, Path)) or hasattr(ik, "columns"):
            kinematics_trajectory = self._ik_to_jnp(
                ik,
                mean_suffix=mean_suffix,
                time_column=time_column,
                lowpass_hz=lowpass_hz,
                sample_rate_hz=sample_rate_hz,
                dt=dt,
            )
        else:
            raise ValueError("Inverse Kinematics input must be a .mot path or DataFrame")


        # Build external loads array from a .mot path or DataFrame
        if isinstance(grf, (str, Path)) or hasattr(grf, "columns"):
            ext_loads_trajectory = self._grf_to_jnp(grf)
        else:
            raise ValueError("GRF input must be a .mot path or DataFrame")

        all_symbols = list(self.all_input_symbols) + list(self.ext_loads_syms) + list(self.offset_syms)
        pelvis_tx_eq = self.substituted_exteqs_.get("pelvis_tx")
        pelvis_ty_eq = self.substituted_exteqs_.get("pelvis_ty")
        pelvis_tilt_eq = self.substituted_exteqs_.get("pelvis_tilt")

        if not all([pelvis_tx_eq, pelvis_ty_eq, pelvis_tilt_eq]):
            raise ValueError("Missing pelvis external-load equations")

        pelvis_tx_fn = sp.lambdify(all_symbols, pelvis_tx_eq, modules="jax", cse=True)
        pelvis_ty_fn = sp.lambdify(all_symbols, pelvis_ty_eq, modules="jax", cse=True)
        pelvis_tilt_fn = sp.lambdify(all_symbols, pelvis_tilt_eq, modules="jax", cse=True)

        num_base_params = len(self.base_params)
        base_params = jnp.zeros(num_base_params)
        offset_values = jnp.asarray(self.offset_defaults) if self.offset_defaults else None

        def pelvis_ext_fn(kinematics, ext_loads):
            if offset_values is None:
                flat_inputs = jnp.concatenate([kinematics, base_params, ext_loads])
            else:
                flat_inputs = jnp.concatenate([kinematics, base_params, ext_loads, offset_values])
            tx = pelvis_tx_fn(*flat_inputs)
            ty = pelvis_ty_fn(*flat_inputs)
            tilt = pelvis_tilt_fn(*flat_inputs)
            return jnp.array([tx, ty, tilt])

        vmap_ext_fn = jax.jit(jax.vmap(pelvis_ext_fn, in_axes=(0, 0)))
        tau_ext_traj = vmap_ext_fn(kinematics_trajectory, ext_loads_trajectory)

        if frame_stride is None:
            if lowpass_hz is not None and lowpass_hz > 0:
                if sample_rate_hz is None:
                    if dt is None:
                        sample_rate_hz = 100.0
                    else:
                        sample_rate_hz = 1.0 / float(dt)
                target_rate = 2.5 * float(lowpass_hz)
                frame_stride = int(np.ceil(float(sample_rate_hz) / target_rate))
        if frame_stride is not None:
            if frame_stride < 1:
                raise ValueError("frame_stride must be >= 1")
            tau_ext_traj = tau_ext_traj[::frame_stride, :]

        tau_ext_stacked = tau_ext_traj.reshape(-1)

        return tau_ext_stacked


def inertial_to_base_2d_mapping(
    model_path: str | None = None,
    model=None,
    base_param_symbols: list[sp.Symbol] | None = None,
    return_defaults: bool = False,
    force_rebuild: bool = False,
    shoe_weight: float = 0.0,
):
    """Build base-parameter mappings for a 2D model.

    The base-parameter order matches ``BiosymModel.base_params["names"]``:
    ``M_total_bp`` followed by ``Izz_bp_*``, ``mcx_bp_*``, and ``mcy_bp_*``
    for each body in ``model.dicts["bodies"]``.

    When ``return_defaults=True``, ``shoe_weight`` is interpreted as the mass of
    one shoe in kg. It is added to each foot mass and foot ``Izz`` without
    changing COM, and the total shoe mass is removed uniformly from the
    remaining body masses.
    """

    def _base_parameter_tree_2d() -> dict[str, str | None]:
        return {
            "foot_r": "tibia_r",
            "tibia_r": "femur_r",
            "femur_r": "pelvis",
            "foot_l": "tibia_l",
            "tibia_l": "femur_l",
            "femur_l": "pelvis",
            "pelvis": None,
        }

    def _validate_base_parameter_model(tree: dict[str, str | None], model) -> None:
        body_names = [body["name"] for body in model.dicts["bodies"]]
        missing = [name for name in tree.keys() if name not in body_names]
        missing_parents = [
            parent
            for parent in tree.values()
            if parent is not None and parent not in body_names
        ]
        if missing or missing_parents:
            missing_all = sorted(set(missing + missing_parents))
            raise ValueError(
                "Base-parameter mapping expects 2D body names, missing: "
                + ", ".join(missing_all)
            )

        required = ["pelvis", "femur_r", "femur_l"]
        missing_required = [name for name in required if name not in body_names]
        if missing_required:
            raise ValueError(
                "Base-parameter mapping requires bodies: " + ", ".join(missing_required)
            )

    def _extract_body_parameters(model, *, defaults: bool) -> dict:
        bodies = {}
        for body_idx, body in enumerate(model.dicts["bodies"]):
            name = body["name"]
            if defaults:
                bodies[name] = {
                    "m": float(model.default_values[model.masses["idx"] + body_idx]),
                    "izz": float(model.default_values[model.inertia["idx"] + body_idx * 6 + 2]),
                    "cx": float(model.default_values[model.com["idx"] + body_idx * 3 + 0]),
                    "cy": float(model.default_values[model.com["idx"] + body_idx * 3 + 1]),
                    "lx": float(model.default_values[model.offset["idx"] + body_idx * 3 + 0]),
                    "ly": float(model.default_values[model.offset["idx"] + body_idx * 3 + 1]),
                }
            else:
                bodies[name] = {
                    "m": model._v[model.masses["idx"] + body_idx],
                    "izz": model._v[model.inertia["idx"] + body_idx * 6 + 2],
                    "cx": model._v[model.com["idx"] + body_idx * 3 + 0],
                    "cy": model._v[model.com["idx"] + body_idx * 3 + 1],
                    "lx": model._v[model.offset["idx"] + body_idx * 3 + 0],
                    "ly": model._v[model.offset["idx"] + body_idx * 3 + 1],
                }
        return bodies

    def _with_shoe_weight(bodies: dict, *, defaults: bool) -> dict:
        shoe_mass = float(shoe_weight or 0.0)
        if shoe_mass == 0.0:
            return bodies
        if shoe_mass < 0.0:
            raise ValueError("shoe_weight must be non-negative.")

        foot_names = [name for name in ("foot_r", "foot_l") if name in bodies]
        if len(foot_names) != 2:
            raise ValueError(
                "shoe_weight requires both 'foot_r' and 'foot_l' in the 2D base-parameter model."
            )

        non_foot_names = [name for name in bodies if name not in foot_names]
        if not non_foot_names:
            raise ValueError("shoe_weight cannot be redistributed without non-foot body segments.")

        mass_to_remove = (shoe_mass * len(foot_names)) / len(non_foot_names)
        adjusted = {name: params.copy() for name, params in bodies.items()}

        for name in non_foot_names:
            new_mass = bodies[name]["m"] - mass_to_remove
            if defaults and new_mass <= 0.0:
                raise ValueError(
                    "shoe_weight is too large to remove uniformly from non-foot segments. "
                    f"Segment would become non-positive: {name}"
                )
            adjusted[name]["m"] = new_mass

        for name in foot_names:
            original_mass = bodies[name]["m"]
            if defaults and original_mass <= 0.0:
                raise ValueError(f"Cannot apply shoe_weight to non-positive foot mass: {name}")
            new_mass = original_mass + shoe_mass
            adjusted[name]["m"] = new_mass
            adjusted[name]["izz"] = bodies[name]["izz"] * (new_mass / original_mass)

        return adjusted

    def _augmented_masses(bodies: dict) -> dict:
        m_aug = {}

        def get_aug_mass(body_name: str):
            if body_name in m_aug:
                return m_aug[body_name]

            mass = bodies[body_name]["m"]
            for child, parent in tree.items():
                if parent == body_name:
                    mass += get_aug_mass(child)

            m_aug[body_name] = mass
            return mass

        for name in bodies.keys():
            get_aug_mass(name)
        return m_aug

    def _transfer_terms(body_name: str, bodies: dict, m_aug: dict):
        transfer_izz = 0
        transfer_mcx = 0
        transfer_mcy = 0

        for child, parent in tree.items():
            if parent == body_name:
                child_m_aug = m_aug[child]
                lx = bodies[child]["lx"]
                ly = bodies[child]["ly"]

                transfer_izz += child_m_aug * (lx**2 + ly**2)
                transfer_mcx += child_m_aug * lx
                transfer_mcy += child_m_aug * ly

        return transfer_izz, transfer_mcx, transfer_mcy

    # This dictionary maps 'child_body' : 'parent_body' . It tells the script how to transfer the masses upwards.
    tree = _base_parameter_tree_2d()

    if model is None:
        if model_path is None:
            raise ValueError("Provide model_path or model instance for base-parameter mapping.")
        from biosym.model.model import load_model

        model = load_model(model_path, force_rebuild=force_rebuild)
    _validate_base_parameter_model(tree, model)

    raw_bodies = _extract_body_parameters(model, defaults=False)
    bodies = raw_bodies
    m_aug = _augmented_masses(bodies)

    # Create Base Parameter mapping dict
    mapping_dict = {}
    if base_param_symbols is None:
        base_params = [sp.Symbol("M_total_bp")]
        for body in model.dicts["bodies"]:
            name = body["name"]
            base_params.extend(
                [
                    sp.Symbol(f"Izz_bp_{name}"),
                    sp.Symbol(f"mcx_bp_{name}"),
                    sp.Symbol(f"mcy_bp_{name}"),
                ]
            )
    else:
        base_params = list(base_param_symbols)
        if len(base_params) != 1 + 3 * len(bodies):
            raise ValueError(
                "base_param_symbols length does not match expected number of base parameters."
            )
    M_total_bp = base_params[0]

    for name, params in bodies.items():
        raw_params = raw_bodies[name]
        if name != "pelvis":
            mapping_dict[raw_params["m"]] = params["m"]

    # Express the old pelvis mass as Total Mass minus all the augmented leg masses
    m_pelvis_expr = M_total_bp - m_aug["femur_r"] - m_aug["femur_l"]
    mapping_dict[raw_bodies["pelvis"]["m"]] = m_pelvis_expr

    # Transfer Inertias and First Moments up the tree
    for body_idx, (name, params) in enumerate(bodies.items()):
        raw_params = raw_bodies[name]
        bp_idx = 1 + 3 * body_idx
        izz_bp = base_params[bp_idx]
        mcx_bp = base_params[bp_idx + 1]
        mcy_bp = base_params[bp_idx + 2]
        transfer_izz, transfer_mcx, transfer_mcy = _transfer_terms(name, bodies, m_aug)

        # If it is the pelvis, we must use our M_total expression so it cancels!
        if name == "pelvis":
            mapped_m = m_pelvis_expr
        else:
            mapped_m = params["m"]

        # Centers of Mass (First moments divided by mass)
        new_cx = (mcx_bp - transfer_mcx) / mapped_m
        new_cy = (mcy_bp - transfer_mcy) / mapped_m
        
        mapping_dict[raw_params["cx"]] = new_cx
        mapping_dict[raw_params["cy"]] = new_cy
        
        # Inertia (mapped_m * (new_cx**2 + new_cy**2) is the inertia of the mass if it were concentrated at the new CoM, so we subtract that out to avoid double counting)
        mapping_dict[raw_params["izz"]] = izz_bp - transfer_izz - mapped_m * (new_cx**2 + new_cy**2)

    if return_defaults:
        default_bodies = _with_shoe_weight(_extract_body_parameters(model, defaults=True), defaults=True)
        default_m_aug = _augmented_masses(default_bodies)
        M_total_default = sum(params["m"] for params in default_bodies.values())
        m_pelvis_default = M_total_default - default_m_aug["femur_r"] - default_m_aug["femur_l"]

        base_param_values = {str(base_params[0]): M_total_default}
        for body_idx, (name, params) in enumerate(default_bodies.items()):
            transfer_izz, transfer_mcx, transfer_mcy = _transfer_terms(
                name, default_bodies, default_m_aug
            )
            mapped_m = m_pelvis_default if name == "pelvis" else params["m"]

            mcx_bp = params["cx"] * mapped_m + transfer_mcx
            mcy_bp = params["cy"] * mapped_m + transfer_mcy
            izz_bp = (
                params["izz"]
                + transfer_izz
                + mapped_m * (params["cx"] ** 2 + params["cy"] ** 2)
            )
            bp_idx = 1 + 3 * body_idx
            base_param_values[str(base_params[bp_idx])] = izz_bp
            base_param_values[str(base_params[bp_idx + 1])] = mcx_bp
            base_param_values[str(base_params[bp_idx + 2])] = mcy_bp

        return mapping_dict, base_params, base_param_values
    return mapping_dict, base_params

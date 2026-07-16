import os
from functools import partial
import jax
import jax.numpy as jnp
from biosym.objectives.base_objective import BaseObjective

class Objective(BaseObjective):
    """
    Objective term for tracking simulated IMU signals against experimental (or synthetic) IMU data.
    """

    def __init__(self, model, settings, **kwargs):
        """
        Initialize the IMU tracking objective function.

        :param model: The loaded BiosymModel.
        :param settings: Collocation settings dict.
        :param tracking_data: Experimental/synthetic IMU data array of shape (n_nodes, n_sensors * 6)
        :param variance_acc: Variance normalization factor for accelerometer tracking.
        :param variance_gyr: Variance normalization factor for gyroscope tracking.
        :param high_freq_weight: Optional weight for high-frequency component (derivative) penalty.
        """
        self.model = model
        self.settings = settings
        self.n_nodes = self.settings["nnodes"]

        # Parse kwargs
        self.tracking_data = kwargs.get("tracking_data", None)
        if self.tracking_data is None:
            # Fallback to zero data if not provided
            self.tracking_data = jnp.zeros((self.n_nodes, 7 * 6))
        else:
            self.tracking_data = jnp.asarray(self.tracking_data)

        self.variance_acc = kwargs.get("variance_acc", 1.0)
        self.variance_gyr = kwargs.get("variance_gyr", 1.0)
        
        self.tracking_variance = kwargs.get("tracking_variance", None)
        if self.tracking_variance is None:
            # Reconstruct time-varying variance array from scalar variance_acc and variance_gyr
            # variance_acc applies to channels 0,1,2 of each of the 7 sensors
            # variance_gyr applies to channels 3,4,5 of each of the 7 sensors
            n_sensors = 7
            var_flat = []
            for _ in range(n_sensors):
                var_flat.extend([self.variance_acc] * 3)
                var_flat.extend([self.variance_gyr] * 3)
            self.tracking_variance = jnp.broadcast_to(jnp.array(var_flat), self.tracking_data.shape)
        else:
            self.tracking_variance = jnp.asarray(self.tracking_variance)

        self.high_freq_weight = kwargs.get("high_freq_weight", 0.0)

        # Instantiate our custom IMU sensor model
        # Import dynamically to handle sys.path injection
        try:
            from sensors import IMUSensorModel
        except ImportError:
            from dorschky_demo.src.sensors import IMUSensorModel
        sensor_config = kwargs.get("sensor_config", None)
        self.sensor_model = IMUSensorModel(self.model, config=sensor_config)

        self.obj_settings = {
            "tracking_data": self.tracking_data,
            "tracking_variance": self.tracking_variance,
            "high_freq_weight": self.high_freq_weight,
            "sensor_model": self.sensor_model,
        }

    def _get_info(self):
        return {
            "name": "IMUTrackingObjective",
            "description": "Objective term for tracking IMU accelerometer and gyroscope signals.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
        }

    def get_objfun(self):
        """Return the objective function."""
        fun = partial(
            objfun,
            settings=self.obj_settings,
            info=self._get_info(),
        )
        return jax.jit(fun)

    def get_gradient(self):
        """Return the gradient of the objective function."""
        fun = partial(
            objfun,
            settings=self.obj_settings,
            info=self._get_info(),
        )
        return jax.jit(jax.grad(fun, argnums=[0, 1]))


def objfun(states_list, globals_dict, settings, info):
    """
    Evaluate the IMU tracking objective function.
    """
    tracking_data = settings["tracking_data"]  # (N, n_sensors * 6)
    tracking_variance = settings["tracking_variance"]  # (N, n_sensors * 6)
    high_freq_weight = settings["high_freq_weight"]
    sensor_model = settings["sensor_model"]

    # Extract state trajectory and constants
    q = states_list.q
    qd = states_list.qd
    constants_model = sensor_model.model.default_constants.filter('model').to_array()

    # Compute qdd using finite difference: qdot(t+1) - qdot(t) / h
    N = qd.shape[0]
    if globals_dict is not None and hasattr(globals_dict, "h") and globals_dict.h is not None and globals_dict.h.size > 0:
        h = globals_dict.h
    elif globals_dict is not None and hasattr(globals_dict, "dur") and globals_dict.dur is not None:
        dt = globals_dict.dur / (N - 1)
        h = jnp.broadcast_to(dt, (N - 1,))
    else:
        h = jnp.ones((N - 1,))
    
    h_clipped = jnp.clip(h, 1e-5, 1.0)[:, None]
    qdd_diff = (qd[1:] - qd[:-1]) / h_clipped
    qdd = jnp.concatenate([qdd_diff, qdd_diff[-1:]], axis=0)

    # Pad tau, ext_forces, ext_torques with zeros
    z = lambda n: jnp.zeros((N, n), dtype=q.dtype)
    tau = z(sensor_model.model.tau.n)
    ef = z(sensor_model.model.ext_forces.n)
    et = z(sensor_model.model.ext_torques.n)

    states_model_batch = jnp.concatenate([q, qd, qdd, tau, ef, et], axis=1)

    # Compute simulated IMU signals
    # shape: (N, n_sensors * 6)
    sim_imu = sensor_model.compute_batch(states_model_batch, constants_model)

    # Element-wise normalized tracking error using time-varying, channel-specific variance
    # Variance is already guaranteed to be >= 1e-5 (or user-defined non-zero scalar) from inputs.
    # Slice tracking arrays to the first nnodes elements to prevent double-weighting the boundary node
    nnodes = info["n_nodes"]
    sim_imu_sliced = sim_imu[:nnodes]
    tracking_data_sliced = tracking_data[:nnodes]
    tracking_variance_sliced = tracking_variance[:nnodes]

    err_tracking = jnp.mean(((sim_imu_sliced - tracking_data_sliced) ** 2) / tracking_variance_sliced)

    total_loss = err_tracking

    # High-frequency derivative penalty (smoothing)
    if high_freq_weight > 0.0:
        # Time steps
        if globals_dict is not None and hasattr(globals_dict, "h") and globals_dict.h is not None and globals_dict.h.size > 0:
            h = globals_dict.h
            h = jnp.clip(h, 1e-5, 1.0)
        elif globals_dict is not None and hasattr(globals_dict, "dur") and globals_dict.dur is not None:
            N = sim_imu.shape[0]
            dt = globals_dict.dur / (N - 1)
            h = jnp.broadcast_to(dt, (N - 1,))
        else:
            h = jnp.ones((sim_imu.shape[0] - 1,))

        # Reshape h to (N-1, 1) for division broadcasting
        h = h.reshape((-1, 1))

        # Time derivative of signals
        diff_imu = (sim_imu[1:] - sim_imu[:-1]) / h
        err_hf = jnp.mean(diff_imu ** 2)
        total_loss = total_loss + high_freq_weight * err_hf

    return total_loss

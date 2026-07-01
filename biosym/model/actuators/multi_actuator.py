"""
Presents several BaseActuator instances as one self.actuators object, so a model
with both coordinate and torque actuators still exposes the single
interface model.py expects. Not a BaseActuator subclass -- it just fans calls out
to its members and concatenates their results.
"""


class MultiActuator:
    def __init__(self, models):
        if not models:
            raise ValueError("MultiActuator requires at least one actuator model.")
        self.models = list(models)

    def get_n_actuators(self):
        return sum(m.get_n_actuators() for m in self.models)

    def get_n_states(self):
        return sum(m.get_n_states() for m in self.models)

    def get_n_constants(self):
        return sum(m.get_n_constants() for m in self.models)

    def get_actuated_joints(self):
        joints = []
        for m in self.models:
            joints.extend(m.get_actuated_joints())
        return joints

    def get_actuators(self):
        merged = {}
        for m in self.models:
            if hasattr(m, "get_actuators"):
                merged.update(m.get_actuators())
        return merged

    def reset(self):
        for m in self.models:
            m.reset()

    def process_eom(self, model):
        for m in self.models:
            m.process_eom(model)

    # forward / load routing across mixed load-types is left for the model.py
    # integration step (it must route each member's output by get_load_type()).
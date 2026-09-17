"""Dataset-backed observations for deployment acceptance without devices."""


class OfflineBridge:
    command_loop_stats = {}

    def __init__(self, root):
        from scripts.OpenpiDeployer.OpenpiDeployer import _offline_observations
        self.observations = _offline_observations(root)

    def ping(self):
        return {"ready": True}

    def configure(self, **settings):
        from scripts.OpenpiDeployer.OpenpiDeployer import OFFLINE_JOINT_LIMITS
        return {**settings, "offline": True, "joint_units": "rad", "gripper_units": "m",
                "gripper_limits": [0.0, 0.1], "joint_limits": OFFLINE_JOINT_LIMITS.tolist()}

    def observation(self):
        return next(self.observations)

    def status(self):
        return {"command_loop": {"offline": True}}

    def stop(self):
        self.observations.close()

    close = stop

"""Frozen finite-torque actuator command; this is not a motion solution."""
import math

def target_velocity(time_seconds):
    # Original action issues step k's command BEFORE advancing [(k-1)*.01,k*.01].
    step = min(1400, max(1, math.floor((float(time_seconds) + 1.e-12) / .01) + 1))
    u = min(1.0, step / 200.0)
    return 1.05 * u * u * (3.0 - 2.0 * u)

def actuator_torque(time_seconds, actual_angular_velocity):
    return max(-45.0, min(45.0, 12.0 * (target_velocity(time_seconds) - actual_angular_velocity)))

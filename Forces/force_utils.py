import numpy as np


def calc_inertial_force(
		center_of_mass: np.ndarray,  # millimeter
		wing_mass: float = 3,  # grams
		dt: float = 1e-4  # sec
) -> np.ndarray:
	"""
	Aside from aerodynamic forces, our wing may also exert an inertial vertical force in z. The reaction force
	(force observed by sensor) in the z	direction will be Fz = – m * z_ddot where z_ddot is the acceleration of the
	centre of mass in z and m is the wing's mass. When acceleration is positive (up), the reaction force on the sensor
	is negative (down). We can break z_ddot down into R * theta_ddot where R = dist from the rotation point to the C.O.M
	"""
	center_of_mass_z = center_of_mass[:, 2]
	center_of_mass_z_dot = np.gradient(center_of_mass_z, dt)
	center_of_mass_z_ddot = np.gradient(center_of_mass_z_dot, dt)
	force = - wing_mass * center_of_mass_z_ddot  # gram * mm / sec^2
	return force / (10**6) # in newtons

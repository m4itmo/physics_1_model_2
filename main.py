import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

r_inner = 5e-2
r_outer = 11e-2
V0 = 4.5e6
L = 19e-2

e = 1.602e-19
m_e = 9.109e-31
r_0 = (r_outer + r_inner) / 2

ln_factor = np.log(r_outer / r_inner)


def equations(t, state, delta_V):
    x, vx, y, vy = state
    Fy = -e * delta_V / ((r_0 + y) * ln_factor)
    return [vx, 0, vy, Fy / m_e]


def solve_motion(delta_V: int):
    #                    [x, vx, y, vy]
    initial_conditions = [0, V0, 0, 0]
    t_span = (0, L / V0)
    t_eval = np.linspace(*t_span, 1000)

    sol = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, args=(delta_V,))

    return sol


def calculate_ay(delta_V, y):
    return -e * delta_V / ((r_0 + y) * ln_factor) / m_e


def find_min_delta_V(delta_V: int = 0, step: int = 10):
    while True:
        delta_V += step
        y = solve_motion(delta_V).y[2]

        if np.max(np.abs(y)) >= (r_outer - r_inner) / 2:
            return delta_V


delta_V_min = find_min_delta_V()
solved_motion = solve_motion(delta_V_min)

t, x, y, vy = solved_motion.t, solved_motion.y[0], solved_motion.y[2], solved_motion.y[3]

ay = calculate_ay(delta_V_min, y)

plt.figure(figsize=(12, 10))

# y(x)
plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.title("Trajectory y(x)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()

# Vy(t)
plt.subplot(2, 2, 2)
plt.plot(t, vy)
plt.title("Radial Velocity Vy(t)")
plt.xlabel("t (s)")
plt.ylabel("Vy (m/s)")
plt.grid()

# ay(t)
plt.subplot(2, 2, 3)
plt.plot(t, ay)
plt.title("Radial Acceleration ay(t)")
plt.xlabel("t (s)")
plt.ylabel("ay (m/s^2)")
plt.grid()

# y(t)
plt.subplot(2, 2, 4)
plt.plot(t, y)
plt.title("Radial Position y(t)")
plt.xlabel("t (s)")
plt.ylabel("y (m)")
plt.grid()

plt.subplots_adjust(hspace=0.25, wspace=0.2)

final_speed = np.sqrt(V0 ** 2 + vy[-1] ** 2)
flight_time = t[-1]

print(f"Minimum Delta V: {delta_V_min:.2f} V")
print(f"Final speed:     {final_speed:.2e} m/s")
print(f"Flight time:     {flight_time:.2e} s")

plt.show()

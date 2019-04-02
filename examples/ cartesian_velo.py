import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time
import openravepy as orpy

ta.setup_logging("INFO")


def main():
    # openrave setup
    env = orpy.Environment()
    env.Load("robots/barrettwam.robot.xml")
    env.SetViewer('qtosg')
    robot = env.GetRobots()[0]

    robot.SetActiveDOFs(range(7))

    # Parameters
    N_samples = 5
    SEED = 9
    dof = 7

    # Random waypoints used to obtain a random geometric path. Here,
    # we use spline interpolation.
    np.random.seed(SEED)
    way_pts = np.random.randn(N_samples, dof) * 0.6
    path = ta.SplineInterpolator(np.linspace(0, 1, 5), way_pts)

    # Create velocity bounds, then velocity constraint object
    vlim_ = robot.GetActiveDOFMaxVel()
    vlim_[robot.GetActiveDOFIndices()] = [80., 80., 80., 80., 80., 80., 80.]
    vlim = np.vstack((-vlim_, vlim_)).T
    # Create acceleration bounds, then acceleration constraint object
    alim_ = robot.GetActiveDOFMaxAccel()
    alim_[robot.GetActiveDOFIndices()] = [800., 800., 800., 800., 800., 800., 800.]
    alim = np.vstack((-alim_, alim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # cartersian velocity
    def cartersian_velocity(q, qd):
        with robot:
            vlim_ = robot.GetDOFVelocityLimits()
            robot.SetDOFVelocityLimits(vlim_ * 1000)  # remove velocity limits to compute stuffs
            robot.SetActiveDOFValues(q)
            robot.SetActiveDOFVelocities(qd)
            velocity_links = robot.GetLinkVelocities()
            robot.SetDOFVelocityLimits(vlim_)
        return velocity_links[6][:3]  # only return the translational components

    F_q = np.zeros((6, 3))
    F_q[:3, :3] = np.eye(3)
    F_q[3:, :3] = -np.eye(3)
    g_q = np.ones(6) * 1
    def F(q):
        return F_q
    def g(q):
        return g_q
    pc_cart_velo = constraint.CanonicalLinearFirstOrderConstraint(
        cartersian_velocity, F, g, 7, discretization_scheme=constraint.DiscretizationType.Interpolation)
    # cartesin velocity finishes

    all_constraints = [pc_vel, pc_cart_velo, pc_acc]

    gridpoints = np.linspace(0, path.get_duration(), 1000)
    instance = algo.TOPPRA(all_constraints, path, gridpoints, solver_wrapper='seidel')

    # Retime the trajectory, only this step is necessary.
    v_srt = 0
    v_end = 0
    t0 = time.time()
    jnt_traj, _ = instance.compute_trajectory(v_srt, v_end)
    print("Parameterization time: {:} secs".format(time.time() - t0))
    ts_sample = np.linspace(0, jnt_traj.get_duration(), 100)
    qs_sample = jnt_traj.eval(ts_sample)
    qds_sample = jnt_traj.evald(ts_sample)
    qdds_sample = jnt_traj.evaldd(ts_sample)

    cart_velo = []
    for q_, qd_ in zip(qs_sample, qds_sample):
        cart_velo.append(cartersian_velocity(q_, qd_))
    cart_velo = np.array(cart_velo)

    plt.plot(ts_sample, cart_velo[:, 0], label="$a_x$")
    plt.plot(ts_sample, cart_velo[:, 1], label="$a_y$")
    plt.plot(ts_sample, cart_velo[:, 2], label="$a_z$")
    plt.plot([ts_sample[0], ts_sample[-1]], [1, 1], "--", c='black', label="Cart. Velo. Limits")
    plt.plot([ts_sample[0], ts_sample[-1]], [-1, -1], "--", c='black')
    plt.xlabel("Time (s)")
    plt.ylabel("Cartesian velocity of the origin of link 6 $(m/s)$")
    plt.legend(loc='upper right')
    plt.show()

    # preview path
    for t in np.arange(0, jnt_traj.get_duration(), 0.01):
        robot.SetActiveDOFValues(jnt_traj.eval(t))
        time.sleep(0.01)  # 5x slow down

    # Compute the feasible sets and the controllable sets for viewing.
    # Note that these steps are not necessary.
    _, sd_vec, _ = instance.compute_parameterization(v_srt, v_end)
    X = instance.compute_feasible_sets()
    K = instance.compute_controllable_sets(v_srt, v_end)

    X = np.sqrt(X)
    K = np.sqrt(K)

    plt.plot(X[:, 0], c='green', label="Feasible sets")
    plt.plot(X[:, 1], c='green')
    plt.plot(K[:, 0], '--', c='red', label="Controllable sets")
    plt.plot(K[:, 1], '--', c='red')
    plt.plot(sd_vec, label="Velocity profile")
    plt.title("Path-position path-velocity plot")
    plt.xlabel("Path position")
    plt.ylabel("Path velocity square")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

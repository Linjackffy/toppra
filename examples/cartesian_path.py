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

    #
    ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(robot, iktype=orpy.IkParameterization.Type.Transform6D)
    if not ikmodel.load():
        ikmodel.autogenerate()

    # Parameters
    N_samples = 5
    SEED = 9
    dof = 7

    # linear geometric path. Here,
    # we use spline interpolation.
    gridSize = 1000;
    pst = np.array([0.3, 0.2, 0.6])
    ped = np.array([-0.3, 0.2, 0.6])
    Tz = orpy.matrixFromAxisAngle([0, 0, 0])
    way_pts = np.zeros((gridSize, dof))
    gridpoints_ = np.linspace(0, 1, gridSize)
    for i in range(0, gridSize):
        point = pst + (ped - pst) * gridpoints_[i]
        Tz[:3, 3] = point
        sol = robot.GetActiveManipulator().FindIKSolution(Tz, orpy.IkFilterOptions.CheckEnvCollisions) # get collision-free solution
        if np.all(sol == None):
            continue
        way_pts[i, :] = sol

    for i in range(1, way_pts.shape[0] - 1):
        for j in range(0, dof):
            if way_pts[i, j] - way_pts[i - 1, j] > np.pi - 1e-5:
                way_pts[i, j] -= np.pi
            elif way_pts[i, j] - way_pts[i - 1, j] < -np.pi + 1e-5:
                way_pts[i, j] += np.pi
    gridpoints = np.linspace(0, 1, way_pts.shape[0])
    path = ta.SplineInterpolator(gridpoints, way_pts)

    # Create velocity bounds, then velocity constraint object
    vlim_ = robot.GetActiveDOFMaxVel() * 2
    # vlim_[robot.GetActiveDOFIndices()] = [80., 80., 80., 80., 80., 80., 80.]
    vlim = np.vstack((-vlim_, vlim_)).T
    # Create acceleration bounds, then acceleration constraint object
    alim_ = robot.GetActiveDOFMaxAccel() * 10
    # alim_[robot.GetActiveDOFIndices()] = [800., 800., 800., 800., 800., 800., 800.]
    alim = np.vstack((-alim_, alim_)).T
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

    # torque limit
    def inv_dyn(q, qd, qdd):
        qdd_full = np.zeros(robot.GetDOF())
        active_dofs = robot.GetActiveDOFIndices()
        with robot:
            # Temporary remove vel/acc constraints
            vlim = robot.GetDOFVelocityLimits()
            alim = robot.GetDOFAccelerationLimits()
            robot.SetDOFVelocityLimits(100 * vlim)
            robot.SetDOFAccelerationLimits(100 * alim)
            # Inverse dynamics
            qdd_full[active_dofs] = qdd
            robot.SetActiveDOFValues(q)
            robot.SetActiveDOFVelocities(qd)
            res = robot.ComputeInverseDynamics(qdd_full)
            # Restore vel/acc constraints
            robot.SetDOFVelocityLimits(vlim)
            robot.SetDOFAccelerationLimits(alim)
        return res[active_dofs]

    tau_max_ = robot.GetDOFTorqueLimits() * 4
    tau_max = np.vstack((-tau_max_[robot.GetActiveDOFIndices()], tau_max_[robot.GetActiveDOFIndices()])).T
    fs_coef = np.random.rand(dof) * 10
    pc_tau = constraint.JointTorqueConstraint(
        inv_dyn, tau_max, fs_coef, discretization_scheme=constraint.DiscretizationType.Interpolation)
    # torque limit finishes

    # setup cartesian velocity constraint to limit link 7
    # -0.5 <= v <= 0.5
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

    cart_vlim = 0.5
    F_q = np.zeros((6, 3))
    F_q[:3, :3] = np.eye(3)
    F_q[3:, :3] = -np.eye(3)
    g_q = np.ones(6) * cart_vlim

    def F(q):
        return F_q

    def g(q):
        return g_q

    pc_cart_velo = constraint.CanonicalLinearFirstOrderConstraint(
        cartersian_velocity, F, g, 7, discretization_scheme=constraint.DiscretizationType.Interpolation)
    # cartesin velocity finishes

    all_constraints = [pc_vel, pc_acc, pc_cart_velo, pc_tau]
    # all_constraints = pc_vel

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

    fig, axs = plt.subplots(dof, 1)
    for i in range(0, robot.GetActiveDOF()):
        axs[i].plot(ts_sample, qds_sample[:, i])
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity $(deg/s)$")
    plt.legend(loc='upper right')
    plt.show()

    torque = []
    for q_, qd_, qdd_ in zip(qs_sample, qds_sample, qdds_sample):
        torque.append(inv_dyn(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
    torque = np.array(torque)

    fig, axs = plt.subplots(dof, 1)
    for i in range(0, robot.GetActiveDOF()):
        axs[i].plot(ts_sample, torque[:, i])
        axs[i].plot([ts_sample[0], ts_sample[-1]], [tau_max[i], tau_max[i]], "--")
        axs[i].plot([ts_sample[0], ts_sample[-1]], [-tau_max[i], -tau_max[i]], "--")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque $(Nm)$")
    plt.legend(loc='upper right')
    plt.show()

    cart_velo = []
    for q_, qd_ in zip(qs_sample, qds_sample):
        cart_velo.append(cartersian_velocity(q_, qd_))
    cart_velo = np.array(cart_velo)

    plt.plot(ts_sample, cart_velo[:, 0], label="$v_x$")
    plt.plot(ts_sample, cart_velo[:, 1], label="$v_y$")
    plt.plot(ts_sample, cart_velo[:, 2], label="$v_z$")
    plt.plot([ts_sample[0], ts_sample[-1]], [cart_vlim, cart_vlim], "--", c='black', label="Cart. Velo. Limits")
    plt.plot([ts_sample[0], ts_sample[-1]], [-cart_vlim, -cart_vlim], "--", c='black')
    plt.xlabel("Time (s)")
    plt.ylabel("Cartesian velocity of the origin of link 6 $(m/s)$")
    plt.legend(loc='upper right')
    plt.show()

    # preview path
    cart_path = []
    for t in np.arange(0, jnt_traj.get_duration(), 0.01):
        robot.SetActiveDOFValues(jnt_traj.eval(t))
        # T6 = robot.GetLinks()[7].GetTransform()
        T6 = robot.GetActiveManipulator().GetEndEffectorTransform()
        cart_path.append(T6[:3, 3])
        time.sleep(0.01)  # 5x slow down
    cart_path = np.array(cart_path)

    fig, axs = plt.subplots(3, 1)
    for i in range(0, 3):
        axs[i].plot(cart_path[:, i])
    plt.show()

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

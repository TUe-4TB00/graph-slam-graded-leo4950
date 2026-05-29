import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate


def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph


def optimize(graph, initial_estimate):
    # Initialize optimizer
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Optimize
    result = optimizer.optimize()

    print(result)
    return result


def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_score = float("inf")

    for pose_key, pose_5 in pose_options.items():

        for landmark in [1, 2]:

            g_tmp = graph.clone()
            v_tmp = gtsam.Values(initial_estimate)

            # Add pose X5
            g_tmp, v_tmp = add_pose(g_tmp, v_tmp, pose_5)
            result = optimize(g_tmp, v_tmp)

            # Add landmark measurement
            g_tmp = add_landmark_measurement(g_tmp, result, pose_5, landmark)
            result = optimize(g_tmp, v_tmp)

            # Compute marginals
            marginals = gtsam.Marginals(g_tmp, result)

            score = 0.0
            for l in [1, 2]:
                cov = marginals.marginalCovariance(L(l))
                score += cov.sum()

            if score < best_score:
                best_score = score
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_score



def minimize_errors(graph, initial_estimate, pose_options):
    # baseline (without X5 changes)
    base_result = optimize(graph, initial_estimate)

    best_pose = None
    best_landmark = None
    best_score = float("inf")

    for pose_key, pose_5 in pose_options.items():

        for landmark in [1, 2]:

            g_tmp = graph.clone()
            v_tmp = gtsam.Values(initial_estimate)

            # Add pose X5
            g_tmp, v_tmp = add_pose(g_tmp, v_tmp, pose_5)
            result = optimize(g_tmp, v_tmp)

            # Add landmark measurement
            g_tmp = add_landmark_measurement(g_tmp, result, pose_5, landmark)
            result = optimize(g_tmp, v_tmp)

            # Compute error on X1–X3
            error = 0.0
            for i in [1, 2, 3]:
                p0 = base_result.atPose2(X(i))
                p1 = result.atPose2(X(i))

                dx = p0.x() - p1.x()
                dy = p0.y() - p1.y()
                dth = p0.theta() - p1.theta()

                error += dx * dx + dy * dy + dth * dth

            if error < best_score:
                best_score = error
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_score
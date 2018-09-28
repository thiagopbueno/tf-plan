import sys
import numpy as np
import tensorflow as tf


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def parse_rddl(path):
    from pyrddl.parser import RDDLParser
    parser = RDDLParser()
    parser.build()
    rddl = parser.parse(read_file(path))
    return rddl


def compile(rddl):
    from rddl2tf.compiler import Compiler
    rddl2tf = Compiler(rddl, batch_mode=True)
    return rddl2tf


def solve(rddl, batch_size, horizon, learning_rate, epochs):
    from tfplan.planners.environment import OnlinePlanning
    from tfplan.planners.online import OnlineOpenLoopPlanner

    # build the compiler
    rddl2tf = compile(rddl)

    # build online planner
    open_loop_planner = OnlineOpenLoopPlanner(rddl2tf, batch_size, horizon, parallel_plans=False)
    open_loop_planner.build(learning_rate, epochs)

    # run plan-execute-monitor cycle and evaluate solution
    planner = OnlinePlanning(rddl2tf, open_loop_planner)
    planner.build()
    trajectories, total_time, avg_time, stddev_time = planner.run(horizon)
    rewards = trajectories[-1]
    total_reward = np.sum(rewards)

    return total_reward, total_time, avg_time, stddev_time


def log_performance(filename, epochs, avg, stddev, minimum, maximum, total_time, avg_time, stddev_time):
    with open(filename, 'a') as file:
        file.write('{},{:.6f},{:.6f},{:.6f},{:.6f},'.format(epochs, avg, stddev, minimum, maximum))
        file.write('{:.4f},{:.4f},{:.4f}\n'.format(total_time, avg_time, stddev_time))


if __name__ == '__main__':
    rddl_file = sys.argv[1]
    output_file = sys.argv[2]

    horizon = 20
    batch_size = 256
    learning_rate = 0.1

    total_epochs = 100
    ratios = [0.10, 0.25]
    N = 2

    print('>> RDDL:          {}'.format(rddl_file))
    print('>> Horizon:       {}'.format(horizon))
    print('>> Batch size:    {}'.format(batch_size))
    print('>> Learning rate: {}'.format(learning_rate))
    print()

    # read RDDL file
    rddl = parse_rddl(rddl_file)

    for ratio in ratios:
        epochs = int(ratio * total_epochs)
        print('>> EPOCHS = {}'.format(epochs))
        print()

        rewards = []
        total_times = []
        avg_times = []
        stddev_times = []
        for n in range(N):
            print('---> n = {}/{}'.format(n+1, N))
            r, total_time, avg_time, stddev_time = solve(rddl, batch_size, horizon, learning_rate, epochs)
            rewards.append(r)
            total_times.append(total_time)
            avg_times.append(avg_time)
            stddev_times.append(stddev_time)
            print('total_reward = {}'.format(r))
            print()
    
        avg_rewards = np.mean(rewards)
        stddev_rewards = np.sqrt(np.var(rewards))
        max_rewards = np.max(rewards)
        min_rewards = np.min(rewards)

        total_time = np.mean(total_times)
        avg_total_time = np.mean(avg_times)
        stddev_total_time = np.mean(stddev_times)

        print()
        print('>> avg = {}, stddev = {}, min = {}, max = {}'.format(avg_rewards, stddev_rewards, min_rewards, max_rewards))
        print()

        log_performance(output_file, epochs, avg_rewards, stddev_rewards, min_rewards, max_rewards,
            total_time, avg_total_time, stddev_total_time)

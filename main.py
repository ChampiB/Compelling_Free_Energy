#
# This main creates an CFE agent and evaluate its performance on the maze solving task.
#

from experiments.Configurations import Configurations as Configs
from environments.MazeEnv import MazeEnv
from experiments.TimeTracker import TimeTracker
from experiments.MazePerformanceTracker import MazePerformanceTracker
from agents.AgentCFE import AgentCFE


def print_progression(f, i, n):
    f.write("========== EXPERIMENT " + str(i + 1) + "/" + str(n) + " ==========\n\n")


def solved(agent_pos, exit_pos):
    return agent_pos[0] == exit_pos[0] and agent_pos[1] == exit_pos[1]


if __name__ == '__main__':

    # The index of the experiment to run.
    MAZE_ID = 1  # 0 -> 1.maze, 1 -> 5.maze, 2 -> 7.maze, 3 -> 8.maze, 4 -> 9.maze, 5 -> 14.maze

    # Create all experiments configurations.
    configs = Configs.create()
    config = configs[MAZE_ID]

    # Open the file in which the result should be written.
    file = open("results/results.txt", "a")

    # Write experiment configuration in the output file.
    print_progression(file, MAZE_ID, len(configs))
    config.print(file)

    # Create environment.
    env = MazeEnv(config.maze_file_name)

    # Create time and performance trackers.
    perf_tracker = MazePerformanceTracker(config.local_minima_pos)
    time_tracker = TimeTracker()

    # Initialise trackers.
    perf_tracker.reset()
    time_tracker.tic()

    # Run the episodes.
    for j in range(config.n_episodes):

        # Reset the environment and create the agent
        o0 = env.reset()
        agent = AgentCFE(env, o0, time_horizon=config.action_perception_cycles)

        # Run one episode.
        env.print()
        for k in range(config.action_perception_cycles):
            agent.step(env)
            env.print()
            # TODO if solved(env.agent_position(), env.exit_position()):
            # TODO     break

        # Evaluate episode.
        perf_tracker.track(env)

    # Print trackers results
    time_tracker.toc()
    time_tracker.print(file)
    perf_tracker.print(file)

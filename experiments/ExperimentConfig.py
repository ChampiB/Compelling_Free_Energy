#
# This class stores the configuration of an experiment.
#
class ExperimentConfig:

    def __init__(self, maze_number, nb_episodes=100, ap_cycles=30):
        local_minima = {
            1:  [[3, 4]],
            5:  [[3, 3]],
            7:  [[8, 3], [4, 7], [1, 3], [4, 1], [6, 3], [4, 5]],
            8:  [[3, 7], [7, 7]],
            9:  [[5, 3], [3, 5]],
            14: [[3, 4], [7, 4]]
        }

        self.local_minima_pos = local_minima[maze_number]
        self.maze_file_name = "./data/mazes/" + str(maze_number) + ".maze"
        self.action_perception_cycles = ap_cycles
        self.n_episodes = nb_episodes

    def print(self, file):
        file.write("========== EXPERIMENT CONFIGURATION ==========\n")
        file.write("Number of action-perception cycles: " + str(self.action_perception_cycles) + "\n")
        file.write("Number of simulations: " + str(self.n_episodes) + "\n")
        file.write("Maze file's name: " + self.maze_file_name + "\n")

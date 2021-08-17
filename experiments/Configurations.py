from experiments.ExperimentConfig import ExperimentConfig as Config


#
# This class create the experiments configurations.
#
class Configurations:

    @staticmethod
    def create():
        return [Config(maze) for maze in [1, 5, 7, 8, 9, 14]]

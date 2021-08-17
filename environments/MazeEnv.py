import numpy as np
from environments.MazeEnvAction import MazeEnvAction


#
# This class load and simulate mazes.
#
class MazeEnv:

    def __init__(self, maze_file_name):
        # Set amount of noise
        self.noise = 0.01

        # Open maze file.
        file = open(maze_file_name, "r")

        # Load maze's size.
        self.maze_size = [int(i) for i in file.readline().split(" ")]
        if len(self.maze_size) != 2:
            raise RuntimeError("Incorrect maze file format: " + maze_file_name + ".")

        # Initialise attributes
        self.agent_pos = [-1, -1]
        self.exit_pos = [-1, -1]
        self.nb_states = 0
        self.maze = np.empty([self.maze_size[0], self.maze_size[1]])

        # Load maze's content.
        for i in range(self.maze_size[0]):
            # Load next line in file.
            line = file.readline()
            for j in range(min(len(line), self.maze_size[1])):
                if line[j] == 'W':
                    self.maze[i][j] = 1
                elif line[j] == '.':
                    self.nb_states += 1
                    self.maze[i][j] = 0
                elif line[j] == 'E':
                    self.nb_states += 1
                    self.maze[i][j] = 0
                    self.exit_pos[0] = i
                    self.exit_pos[1] = j
                elif line[j] == 'S':
                    self.nb_states += 1
                    self.maze[i][j] = 0
                    self.agent_pos[0] = i
                    self.agent_pos[1] = j
                else:
                    raise RuntimeError("Incorrect maze file format: " + maze_file_name + ".")
            # Add walls for incomplete lines.
            for j in range(len(line), self.maze_size[1]):
                self.maze[i][j] = 1

        # Remember the initial position of the agent.
        self.agent_initial_pos = self.agent_pos.copy()

        # Load state indices.
        self.states_ids = self.load_states_indices()

    def load_states_indices(self):
        state_id = 0
        states_ids = np.full(self.maze.shape, -1)

        for j in range(self.maze.shape[0]):
            for i in range(self.maze.shape[1]):
                if self.maze[j][i] == 0:
                    states_ids[j][i] = state_id
                    state_id += 1
        return states_ids

    def reset(self):
        self.agent_pos = self.agent_initial_pos.copy()
        return self.execute(MazeEnvAction.IDLE)

    def execute(self, action):
        self.agent_pos = self.execute_in_position(action, self.agent_pos)
        return MazeEnv.manhattan_distance(self.agent_pos, self.exit_pos)

    def execute_in_position(self, action, pos):
        res = pos.copy()
        if action == MazeEnvAction.UP:
            if res[0] - 1 >= 0 and self.maze[res[0] - 1][res[1]] == 0:
                res[0] -= 1
        elif action == MazeEnvAction.DOWN:
            if res[0] + 1 < self.maze.shape[0] and self.maze[res[0] + 1][res[1]] == 0:
                res[0] += 1
        elif action == MazeEnvAction.LEFT:
            if res[1] - 1 >= 0 and self.maze[res[0]][res[1] - 1] == 0:
                res[1] -= 1
        elif action == MazeEnvAction.RIGHT:
            if res[1] + 1 < self.maze.shape[1] and self.maze[res[0]][res[1] + 1] == 0:
                res[1] += 1
        elif action != MazeEnvAction.IDLE:
            raise RuntimeError("Invalid action was sent to MazeEnv.execute.")
        return res

    def print(self):
        for i in range(self.maze.shape[0]):
            for j in range(0, self.maze.shape[1]):
                if self.agent_pos[0] == i and self.agent_pos[1] == j:
                    print("A", end="")
                elif self.exit_pos[0] == i and self.exit_pos[1] == j:
                    print("E", end="")
                elif self.maze[i][j] == 0:
                    print(" ", end="")
                else:
                    print("W", end="")
            print()
        print("A = agent position")
        print("E = exit position")
        print("W = wall")

    def agent_position(self):
        return self.agent_pos

    def exit_position(self):
        return self.exit_pos

    def a(self):
        a_mat = np.full([self.observations(), self.states()], self.noise / (self.observations() - 1))

        for i in range(self.maze.shape[1]):
            for j in range(self.maze.shape[0]):
                if self.maze[j][i] == 0:
                    dist = self.manhattan_distance([j, i], self.exit_pos)
                    a_mat[dist][self.states_ids[j][i]] = 1 - self.noise
        return a_mat

    def b(self):
        b_mat = np.full([self.states(), self.states(), self.actions()], self.noise / (self.states() - 1))

        for i in range(self.maze.shape[1]):
            for j in range(self.maze.shape[0]):
                if self.maze[j][i] == 0:
                    current_pos = [j, i]
                    for k in range(self.actions()):
                        pos = self.execute_in_position(k, current_pos)
                        b_mat[self.states_ids[pos[0]][pos[1]]][self.states_ids[j][i]][k] = 1 - self.noise
        return b_mat

    def d(self):
        d_mat = np.full([self.states()], self.noise / (self.states() - 1))
        d_mat[self.states_ids[self.agent_pos[0]][self.agent_pos[1]]] = 1 - self.noise
        return d_mat

    @staticmethod
    def actions():
        return 5

    def states(self):
        return self.nb_states

    def observations(self):
        return self.maze.shape[0] + self.maze.shape[1] - 5

    @staticmethod
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

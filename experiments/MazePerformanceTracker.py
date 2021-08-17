class MazePerformanceTracker:

    def __init__(self, local_minima_pos, tolerance_level = 1):
        self.local_pos = local_minima_pos
        self.tolerance = tolerance_level
        # Reserved space for number of local minima + global minimum + other
        self.perf = [0] * (len(local_minima_pos) + 2)

    def reset(self):
        for i in range(len(self.perf)):
            self.perf[i] = 0

    def track(self, env):
        agent_pos = env.agent_position()
        exit_pos = env.exit_position()
        local_min = -1

        for i in range(len(self.local_pos)):
            if env.manhattan_distance(agent_pos, self.local_pos[i]) <= self.tolerance:
                local_min = i

        if env.manhattan_distance(agent_pos, exit_pos) <= self.tolerance:
            self.perf[len(self.perf) - 1] += 1
        elif local_min != - 1:
            self.perf[local_min + 1] += 1
        else:
            self.perf[0] += 1

    def print(self, file):
        total = sum(self.perf)
        file.write("========== MAZE PERFORMANCE TRACKER ==========\n")
        file.write("P(global): " + str(self.perf[len(self.perf) - 1] / total) + "\n")
        for i in range(len(self.local_pos)):
            file.write("P(local " + str(i + 1) + "): " + str(self.perf[i + 1] / total) + "\n")
        file.write("P(other): " + str(self.perf[0] / total) + "\n\n")

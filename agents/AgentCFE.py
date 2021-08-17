from gekko import GEKKO
from scipy.special import softmax
import numpy as np
from operators.Operators import Operators as Ops
from environments.MazeEnvAction import MazeEnvAction


#
# This class emulate the CFE agent.
#
class AgentCFE:

    def __init__(self, env, obs, time_horizon=30):
        # Sanity check
        if time_horizon < 1:
            raise RuntimeError("CFE::CFE the time horizon must be at least one.")

        # Time Horizon
        self.T = time_horizon

        # Convergence threshold
        self.epsilon = 0.01

        # Prior parameters
        self.a = env.a()
        self.b = env.b()
        self.d = env.d()

        # Shared parameters
        self.e = []
        for i in range(self.T):
            self.e.append(Ops.uniform([env.actions()]))

        # Evidence
        self.o = []
        self.o.append(Ops.one_hot(env.observations(), obs))

        # Prior preferences
        self.c = Ops.one_hot(env.observations(), 0)
        # OR self.c = softmax(env.observations() - np.arange(0, env.observations()), axis=0)

        # Posterior parameters
        self.b_hat = []
        for i in range(self.T):
            self.b_hat.append(Ops.uniform([env.states(), env.actions()]))
        self.d_hat = Ops.uniform([env.states()])

    def step(self, env):
        self.inference()
        action = self.action_selection()
        obs = env.execute(action)
        self.o.append(Ops.one_hot(env.observations(), obs))

    def inference(self):
        cfe = float("inf")
        actions = self.e[0].size
        # bad_actions = [MazeEnvAction.UP] * self.T
        # good_actions_maze_5 = \
        #     [MazeEnvAction.LEFT, MazeEnvAction.UP, MazeEnvAction.UP, MazeEnvAction.RIGHT] + \
        #     [MazeEnvAction.IDLE] * (self.T - 4)
        # good_actions_maze_1 = \
        #     [MazeEnvAction.UP] * 4 + [MazeEnvAction.RIGHT] * (self.T - 4)

        while True:
            # Optimise parameters of the variational and compelled distributions.
            self.update_posterior_over_hidden_states(actions)
            self.update_posterior_over_actions(actions, "LinProg")
            # OR self.update_posterior_over_actions(actions_seq, "Fixed")

            # Check convergence of the CFE
            new_cfe = self.cfe()
            if cfe - new_cfe < self.epsilon:
                break
            cfe = new_cfe

        print("cfe:" + str(cfe))

    def update_posterior_over_hidden_states(self, actions):
        # Inference of initial hidden state
        s = np.zeros(self.d_hat.shape)
        s += np.matmul(np.log(self.a).T, self.z(0))
        s += np.log(self.d)
        s += Ops.average(np.log(self.b), Ops.multiplication(self.b_hat[1], self.e[0], [1]), [0, 2])
        self.d_hat = softmax(s, 0)

        # Inference of hidden states (tau > 0)
        for i in range(self.T):
            s = np.zeros(self.b_hat[i].shape)
            s += Ops.expansion(np.matmul(np.log(self.a).T, self.z(i + 1)), actions, 1)
            if i + 1 != self.T:
                tmp = Ops.average(np.log(self.b), Ops.multiplication(self.b_hat[i + 1], self.e[i + 1], [1]), [0, 2])
                s += Ops.expansion(tmp, actions, 1)
            s += Ops.average(np.log(self.b), self.get_d_hat(i), [1])
            self.b_hat[i] = softmax(s, 0)

    def z(self, tau):
        if len(self.o) <= tau < self.T - 1:
            return np.zeros(self.c.shape)
        return self.c if tau >= len(self.o) else self.o[tau]

    def get_d_hat(self, tau):
        return self.d_hat if tau == 0 else np.matmul(self.b_hat[tau - 1], self.e[tau - 1])

    def update_posterior_over_actions(self, actions, update_type):
        if update_type == "LinProg":
            self.update_posterior_over_actions_using_lp(actions)
        if update_type == "Fixed":
            self.update_posterior_over_actions_using_f(actions)

    def update_posterior_over_actions_using_f(self, actions):
        # Iterate over all actions
        for tau in range(self.T):
            self.e[tau] = Ops.one_hot(self.e[tau].size, actions[tau])

    def update_posterior_over_actions_using_lp(self, actions):
        # Iterate over all actions
        for tau in range(self.T):

            # Create the linear programming solver
            solver = GEKKO(remote=False)

            # Create one variables for each parameter of R(U_tau)
            variables = solver.Array(solver.Var, [actions])
            for i in range(actions):
                variables[i].lower = 0
                variables[i].upper = 1

            # Constrain the parameters of R(U_tau) to remain on the simplex
            solver.Equation(solver.sum(variables) == 1)

            # Create the objective function
            w = self.compute_linear_programming_weights(tau)
            solver.qobj(w, x=variables, otype="min")

            # Compute the optimal parameter of R(U_tau)
            solver.solve(disp=False)

            # Retrieve the results
            for i in range(actions):
                self.e[tau][i] = variables[i].value[0]

    def compute_linear_programming_weights(self, tau):
        s = np.zeros(self.b_hat[0].shape)

        s += Ops.expansion(- np.matmul(np.log(self.a).T, self.z(tau + 1)), s.shape[1], 1)
        s += np.log(self.b_hat[tau])
        s += - Ops.average(np.log(self.b), self.get_d_hat(tau + 1), [1])
        return Ops.average(s, self.b_hat[tau], [0, 1], [1])

    def cfe(self, with_constant=False):
        fe = 0

        # Compute accuracy and expected disappointment
        for tau in range(self.T):
            fe -= np.inner(np.matmul(np.log(self.a).T, self.z(tau)), self.get_d_hat(tau))
            if tau >= len(self.o) and with_constant:
                fe = np.inner(self.c, np.log(self.c))

        # Compute complexity over initial states
        fe += np.inner(np.log(self.d_hat) - np.log(self.d), self.d_hat)

        # Compute complexity over non-initial states
        for tau in range(self.T):
            diff = np.log(self.b_hat[tau]) - Ops.average(np.log(self.b), self.get_d_hat(tau), [1])
            joint = Ops.multiplication(self.b_hat[tau], self.e[tau], [1])
            fe += Ops.average(diff, joint, [0, 1])
        return fe

    def print_posterior(self):
        print("=====> Q(S_0)")
        print(np.around(self.d_hat, decimals=2))
        for i in range(self.T):
            print("=====> Q(U_" + str(i) + ")")
            print(np.around(self.e[i], decimals=2))
            print("=====> Q(S_" + str(i + 1) + "|U_" + str(i) + ")")
            print(np.around(self.b_hat[i], decimals=2))

    def print_posterior_over_hidden_states(self):
        print("===== Posterior over hidden states =====")
        print("=====> Q(S_0)")
        print(np.around(self.d_hat, decimals=2))
        for i in range(self.T):
            print("=====> Q(S_" + str(i + 1) + "|U_" + str(i) + ")")
            print(np.around(self.b_hat[i], decimals=2))

    def print_posterior_over_actions(self):
        print("===== Posterior over actions =====")
        for i in range(self.T):
            print("=====> Q(U_" + str(i) + ")")
            print(np.around(self.e[i], decimals=2))

    def action_selection(self):
        p_actions = self.e[len(self.o) - 1]
        n_actions = p_actions.size
        return np.random.choice(np.arange(n_actions), p=p_actions)

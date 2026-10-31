import math


class RewardCalculator:
    """
    Reward functions for the multi-objective (energy vs. utilization) environment.

    reward_f:
        "G"  : makespan (sparse, terminal only)
        "J"  : just-in-time (penalize tokens waiting in machine buffers)
        "E"  : energy only
        "MU" : machine utilization only
        "MX" : scalarized mix of energy and utilization, weighted by usage_cap

    scalarization (only used by "MX"):
        "linear"    : Wu * U + We * E with the shaped cosine rewards (thesis Eq. 4.1-4.3)
        "chebyshev" : -max(We * energy_deviation, Wu * (1 - utilization))
    """

    TERMINAL_BONUS = 50

    def __init__(self, env, scalarization="linear"):
        if scalarization not in ("linear", "chebyshev"):
            raise ValueError("scalarization must be 'linear' or 'chebyshev'")
        self.env = env
        self.scalarization = scalarization
        self.reward_function = self.get_reward_function()

    def get_reward_function(self):
        reward_functions = {
            "G": self.general_reward,
            "J": self.jit_reward,
            "E": self.energy_reward,
            "MX": self.reward_mix,
            "MU": self.machine_utilization,
        }
        return reward_functions.get(self.env.reward_f, lambda f, t: 0)

    def machine_utilization(self, feedback, terminal):
        """Shaped utilization reward: linear above 90 %, cosine ramp below (Eq. 4.1)."""
        machine_util = feedback["machine_utilisation"]
        if machine_util >= 0.9:
            return machine_util
        return -math.cos(3 * (machine_util + 2.14))

    def energy_reward(self, feedback, terminal):
        """Shaped energy reward: 1 under 10 % of total capacity, cosine decay above (Eq. 4.2)."""
        consumption = self.env.sim.feedbacks["energy_consumption"]
        if consumption == 0:
            return 0

        total_power = self.env.sim.machines_powers.sum()
        if consumption <= 0.1 * total_power:
            return 1
        return math.cos(3 * ((consumption / total_power) + 2.14))

    def energy_deviation(self):
        """Distance of the current consumption from the 10 % target, normalized to [0, 1]."""
        consumption = self.env.sim.feedbacks["energy_consumption"]
        total_power = self.env.sim.machines_powers.sum()
        return abs(consumption - 0.1 * total_power) / total_power

    def reward_mix(self, feedback, terminal):
        if terminal:
            return self.TERMINAL_BONUS

        weight_util = self.env.sim.usage_cap
        weight_energy = 1 - weight_util

        if self.scalarization == "linear":
            energy_r = self.energy_reward(feedback, terminal)
            util_r = self.machine_utilization(feedback, terminal)
            return weight_util * util_r + weight_energy * energy_r

        # Chebyshev: minimize the largest weighted regret of both objectives
        util_regret = 1 - feedback["machine_utilisation"]
        return -max(weight_energy * self.energy_deviation(), weight_util * util_regret)

    def general_reward(self, feedback, terminal):
        if terminal:
            self.env.min_makespan = min(self.env.min_makespan, self.env.sim.clock)
            return -self.env.sim.clock
        return 0

    def jit_reward(self, feedback, terminal):
        total_tokens_in_buffer = sum(len(buffer.token_container)
            for buffer in self.env.sim.places.values()
            if buffer.uid in self.env.sim.filter_nodes("machine_buffer")
        )
        return -total_tokens_in_buffer

    def calculate(self, feedback=None, terminal=False):
        """
        Compute the reward using the predefined reward function.
        """
        return self.reward_function(feedback, terminal)

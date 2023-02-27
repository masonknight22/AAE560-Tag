import mesa
import numpy as np
from strategies import direct_approach, direct_flee
import matplotlib.pyplot as plt
from visualization import GifFactory


class TagModel(mesa.Model):

    num_agents = None
    agent_ids = None
    agent_pos = None
    agent_status = None
    do_plot = None
    bound_rad = None
    gif_maker = None
    _circle_dat = None
    assimilation_dist = None

    def __init__(self, _num_runners, _num_chasers, do_plot=False, bound_rad=50.0):

        # Initialize Model
        super().__init__()
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.num_agents = _num_runners + _num_chasers
        self.do_plot = do_plot
        self.bound_rad = bound_rad
        self.agent_status = np.zeros(self.num_agents, dtype=np.bool_)
        self.agent_pos = np.zeros([self.num_agents, 2])
        self.assimilation_dist = 2.0  # [m]

        # # Create Agents
        # Create runners Agents
        for i in range(_num_runners):
            _runners = BaseAgent(i, self, is_chaser=False, bound_rad=self.bound_rad)
            self.schedule.add(_runners)
        # Create chaser Agents
        for i in range(_num_runners, _num_runners + _num_chasers):
            _chaser = BaseAgent(i, self, is_chaser=True, bound_rad=self.bound_rad)
            self.schedule.add(_chaser)

        # Copy Unique Agent IDs
        self.agent_ids = np.array(range(self.num_agents))

        # Preload agent positions and chaser status
        self.update_agent_status()

        # Visualization Code
        if self.do_plot:
            self.gif_maker = GifFactory("./frames/.gif")
            self.plot_status()

    def step(self) -> bool:  # Model Step
        # If all agents are chasers, end simulation
        if self.agent_status.all():
            return True

        # Perform Agent Steps
        self.schedule.step()

        # Update Agent Status and Check for assimilation
        self.update_agent_status()

        # Visualization Code
        if self.do_plot:
            self.plot_status()

        return False  # Return False; some agents are still runners

    def update_agent_status(self):
        # Update Agent Model Positions and chaser Status

        # Copy Position and chaser Status
        for i, agent in enumerate(self.agents):
            self.agent_pos[i, :] = agent.position.copy()
            self.agent_status[i] = agent.is_chaser

        # Check for runners that will be assimilated
        for i, agent in enumerate(self.agents):
            if not agent.is_chaser:  # If the agent is not a chaser
                nearest_idx, nearest_dist = self.agents[i].nearest_other_team()
                if nearest_dist <= self.assimilation_dist:  # And the agent is close enough to a chaser
                    # Then the agent becomes a chaser
                    self.agents[i].is_chaser = True
                    self.agent_status[i] = True
        return

    def plot_status(self):
        # Plot agent positions and the boundary
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.circle_dat[:, 0], self.circle_dat[:, 1], color="blue")
        plt.title("Tag")
        mk = "o"
        for i in range(len(self.agent_ids)):
            if self.agent_status[i]:  # chaser
                plt.plot(self.agent_pos[i, 0], self.agent_pos[i, 1], color="green", marker=mk)
            else:
                plt.plot(self.agent_pos[i, 0], self.agent_pos[i, 1], color="red", marker=mk)
        plt.xlim([-self.bound_rad, self.bound_rad])
        plt.ylim([-self.bound_rad, self.bound_rad])
        self.gif_maker.append_to_gif(fig)

    def save_gif(self):
        # Save the generated gif
        if self.do_plot:
            self.gif_maker.save_gif()

    @property
    def circle_dat(self):
        # Boundary Calculation
        if self._circle_dat is None:  # Not Calculated Yet
            res = 100
            ang_vec = np.linspace(0, 2*np.pi, res)
            self._circle_dat = np.zeros([res, 2])
            self._circle_dat[:, 0] = np.cos(ang_vec) * self.bound_rad
            self._circle_dat[:, 1] = np.sin(ang_vec) * self.bound_rad
            return self._circle_dat
        else:  # Return Pre-Calculated Circle
            return self._circle_dat

    @property
    def agents(self):
        # Alias for self.schedule.agents
        return self.schedule.agents


class BaseAgent(mesa.Agent):

    is_chaser = None
    run_speed = 3.5  # m/s
    run_variation = 0  # m/s
    walk_speed = 1.42  # m/s
    perception_dist = 20  # m
    _target = None
    pursuit_strategy = None
    position = None
    model: TagModel = None
    vel = None
    timestep = 0.5  # s
    bound_rad = None

    def __init__(self, unique_id, model: TagModel, is_chaser, bound_rad=50.0):
        super().__init__(unique_id, model)
        self.is_chaser = is_chaser
        self.bound_rad = bound_rad
        self.run_speed += np.random.random(1) * self.run_variation - self.run_variation/2

        # Set initial position randomly
        self.position = (np.random.random(2) - 0.5) * 2 * self.bound_rad / np.sqrt(2)

    def step(self) -> None:
        if self.is_chaser:  # chaser Decision Tree

            # Check if target is valid
            if self.target is None:  # If no target, select the nearest runner
                targ_id, targ_dist = self.nearest_other_team()
                self.target = targ_id
            if self.target.is_chaser:  # If target is chaser, select new target
                targ_id, targ_dist = self.nearest_other_team()
                self.target = targ_id

            # Determine runner Position
            target_pos = self.target.position

            # Decide Move Direction
            direction = direct_approach(self.position, target_pos)  # , ally_pos)
            self.vel = self.run_speed * direction  # Calculate Velocity

        else:  # runner Decision Tree

            # Detect chaser in perception range
            visible_chaser, nearest_chaser = self.visible_entity(is_ally=False)

            # If chaser in perception range
            if len(visible_chaser) > 0:
                # Determine Run Direction
                direction = direct_flee(self.position,
                                        self.model.agent_pos[nearest_chaser])
                self.vel = self.run_speed * direction
            else:
                # Select Random Wander Direction
                direction = np.random.random(2)*2-1
                direction = direction / np.linalg.norm(direction)
                self.vel = self.walk_speed * direction
        return

    def advance(self) -> None:
        # Update Position
        self.position += self.vel * self.timestep

        # If out of bounds, restrict to in bounds
        pos_norm = np.linalg.norm(self.position)
        if pos_norm > self.bound_rad:
            self.position = self.position * self.bound_rad / pos_norm
        return

    def nearest_other_team(self):
        if self.is_chaser:
            on_other_team = np.invert(self.model.agent_status)
        else:
            on_other_team = self.model.agent_status
        other_team_pos = self.model.agent_pos[on_other_team]
        other_team_ids = self.model.agent_ids[on_other_team]
        other_team_dist = np.linalg.norm(other_team_pos-self.position, axis=1)
        min_idx = np.argmin(other_team_dist)
        nearest_dist = other_team_dist[min_idx]
        return other_team_ids[min_idx], nearest_dist

    def visible_entity(self, is_ally):
        if (is_ally and self.is_chaser) or (not is_ally and not self.is_chaser):
            search_entity = self.model.agent_status
        else:
            search_entity = np.invert(self.model.agent_status)
        search_entity[self.unique_id] = False
        entity_pos = self.model.agent_pos[search_entity]
        entity_ids = self.model.agent_ids[search_entity]
        entity_dist = np.linalg.norm(entity_pos - self.position, axis=1)
        close_entity_mask = entity_dist <= self.perception_dist
        entity_dist = entity_dist[close_entity_mask]
        entity_ids = entity_ids[close_entity_mask]
        if len(entity_dist) > 0:
            nearest_entity_idx = np.argmin(entity_dist)
            nearest_entity = entity_ids[nearest_entity_idx]
        else:
            nearest_entity = []
        return entity_ids, nearest_entity

    @property
    def target(self):
        if self._target is None:
            return None
        else:
            return self.model.schedule.agents[self._target]

    @target.setter
    def target(self, nearest_agent_id):
        self._target = nearest_agent_id


if __name__ == "__main__":
    # Initialize Constants
    np.random.seed(1000)
    num_runnerss = 17
    num_chaser = 3
    map_bounds = 25.0
    chaser_sim = TagModel(num_runnerss, num_chaser, do_plot=True, bound_rad=map_bounds)

    # Step Through the Simulation
    max_step = 500
    for step_ind in range(max_step):
        finished = chaser_sim.step()
        if finished:
            break
        print("Step " + str(step_ind+1) + "/" + str(max_step))
    chaser_sim.save_gif()

    print("Fin")

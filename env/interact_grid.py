import numpy as np
from gym_minigrid.minigrid import MiniGridEnv, WorldObj, COLORS, fill_coords, point_in_rect, point_in_circle

# COLOR_CYCLE = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
COLOR_CYCLE = ['red', 'blue']
COLOR_NOISE = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

class IKey(WorldObj):
    def __init__(self, color='blue'):
        super(IKey, self).__init__('key', color)

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        """
        Color change logic: cycle through colors in COLOR_CYCLE order
        """
        if self.color in COLOR_CYCLE:
            current_idx = COLOR_CYCLE.index(self.color)
            next_idx = (current_idx + 1) % len(COLOR_CYCLE)
            self.color = COLOR_CYCLE[next_idx]
        else:
            self.color = COLOR_CYCLE[0]
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class IBall(WorldObj):
    def __init__(self, color='blue'):
        super(IBall, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        """
        Color change logic: cycle through colors in COLOR_CYCLE order
        """
        if self.color in COLOR_CYCLE:
            current_idx = COLOR_CYCLE.index(self.color)
            next_idx = (current_idx + 1) % len(COLOR_CYCLE)
            self.color = COLOR_CYCLE[next_idx]
        else:
            self.color = COLOR_CYCLE[0]
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class IBox(WorldObj):
    def __init__(self, color, contains=None):
        super(IBox, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        """
        Implement swap logic:
        1. Temporarily store what the Agent is holding (env.carrying)
        2. Replace what the Agent is holding with what's in the Box (self.contains)
        3. Replace what's in the Box with what was temporarily stored
        """
        obj_in_hand = env.carrying
        obj_in_box = self.contains

        env.carrying = obj_in_box
        self.contains = obj_in_hand

        if env.carrying is not None:
            env.carrying.cur_pos = np.array([-1, -1])

        return True
    
    # Color Change (Use this toggle logic when collecting random playing data)
    # def toggle(self, env, pos):
    #     """
    #     Color change logic: cycle through colors in COLOR_CYCLE order
    #     """
    #     if self.color in COLOR_CYCLE:
    #         current_idx = COLOR_CYCLE.index(self.color)
    #         next_idx = (current_idx + 1) % len(COLOR_CYCLE)
    #         self.color = COLOR_CYCLE[next_idx]
    #     else:
    #         self.color = COLOR_CYCLE[0]
    #     return True


class NoiseFloor(WorldObj):
    """Background noise object: agent can overlap, not pickup-able, renders as colored tile."""
    def __init__(self, color='blue'):
        # Use 'floor' type so MiniGrid encoding does not raise
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c)


class InteractiveMiniGridEnv(MiniGridEnv):
    """
    This is an enhanced MiniGrid environment base class.
    It redefines the originally unused 'done' action as 'switch' (swap objects in hand and in front).
    """

    def step(self, action):
        # Let the parent class run standard logic first (handles step counter and standard actions)
        # If action is done, the parent class does nothing, just consumes one step of time
        obs, reward, done, info = super().step(action)

        # Add custom Switch logic
        if action == self.Actions.done:
            # Get the position and object in front (parent class doesn't pass these variables out)
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if self.carrying is not None and fwd_cell is not None:
                # Ensure the object in front can be picked up (avoid swapping things into walls or doors)
                if fwd_cell.can_pickup():
                    temp_new_obj = fwd_cell
                    self.grid.set(*fwd_pos, self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = temp_new_obj
                    self.carrying.cur_pos = np.array([-1, -1])
                    
                    # Critical: regenerate observation since we changed the environment state
                    # Otherwise the Agent will only see the changes in the next step
                    obs = self.gen_obs()

        return obs, reward, done, info
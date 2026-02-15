"""
Expert policies for MiniGrid environment data collection.
"""

import numpy as np
import random
from collections import deque
from constants import (
    ID_EMPTY, ID_WALL, ID_KEY, ID_BALL, ID_BOX, ID_GOAL, ID_AGENT,
    ACT_LEFT, ACT_RIGHT, ACT_FORWARD, ACT_PICKUP, ACT_DROP, ACT_TOGGLE, ACT_DONE
)


def get_coords(grid_obj_layer, obj_id):
    """
    Extract object coordinates (y, x) from Channel 0.
    Note: numpy returns (row, col), corresponding to MiniGrid's (y, x).
    """
    coords = np.argwhere(grid_obj_layer == obj_id)
    if len(coords) > 0:
        return coords[0]
    return None


def find_empty_position_near_box(obj_layer, col_layer, box_pos, goal_pos, width, height):
    """
    Find an empty position adjacent to the box (for dropping ball).
    Requirements:
    1. Position must be empty (ID_EMPTY)
    2. Cannot be the goal position
    3. Must be adjacent to the box (up, down, left, right)
    
    Returns: (y, x) or None
    """
    if box_pos is None:
        return None
    
    box_y, box_x = box_pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dy, dx in directions:
        ny, nx = box_y + dy, box_x + dx
        if 0 <= ny < height and 0 <= nx < width:
            obj_id = obj_layer[ny, nx]
            if obj_id == ID_EMPTY:
                if goal_pos is None or (ny, nx) != (goal_pos[0], goal_pos[1]):
                    return (ny, nx)
    
    return None


class TensorBFS:
    """
    Path finder based on tensor observations.
    """
    def __init__(self, grid_obj_layer, width, height):
        self.grid = grid_obj_layer
        self.width = width
        self.height = height

    def get_path(self, start_yx, start_dir, target_yx):
        """
        Compute shortest action sequence from start to adjacent position of target.
        start_yx: (y, x)
        start_dir: 0=Right, 1=Down, 2=Left, 3=Up (MiniGrid standard)
        target_yx: (y, x)
        """
        queue = deque([(start_yx[0], start_yx[1], start_dir, [])])
        visited = set([(start_yx[0], start_yx[1], start_dir)])
        
        target_y, target_x = target_yx

        while queue:
            y, x, d, actions = queue.popleft()

            dy, dx = [(1, 0), (0, 1), (-1, 0), (0, -1)][d]
            front_y, front_x = y + dy, x + dx

            if (front_y, front_x) == (target_y, target_x):
                return actions

            for act, new_d in [(ACT_LEFT, (d - 1) % 4), (ACT_RIGHT, (d + 1) % 4)]:
                if (y, x, new_d) not in visited:
                    visited.add((y, x, new_d))
                    queue.append((y, x, new_d, actions + [act]))

            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                obj_id = self.grid[ny, nx]
                is_walkable = (obj_id == ID_EMPTY) or (obj_id == ID_GOAL) or (obj_id == ID_AGENT)
                
                if is_walkable:
                    if (ny, nx, d) not in visited:
                        visited.add((ny, nx, d))
                        queue.append((ny, nx, d, actions + [ACT_FORWARD]))
        
        return None


class BaseExpertPolicy:
    """Base class for expert policies."""
    def __init__(self):
        self.plan = []
        self.finder = None
        self.width = 0
        self.height = 0

    def init_grid_info(self, obs_tensor):
        """Parse grid information."""
        if len(obs_tensor.shape) == 4:
            grid = obs_tensor[0]
        else:
            grid = obs_tensor
        
        self.h, self.w, _ = grid.shape
        self.obj_layer = grid[:, :, 0]
        self.col_layer = grid[:, :, 1]
        
        self.finder = TensorBFS(self.obj_layer, self.w, self.h)
        
        agent_pos = get_coords(self.obj_layer, ID_AGENT)
        if agent_pos is not None:
            agent_dir = grid[agent_pos[0], agent_pos[1], 2]
        else:
            raise ValueError("Agent position not found")
        return grid, agent_pos, agent_dir

    def move_to_target(self, agent_pos, agent_dir, target_pos, actions=None):
        """Generate plan to move to target."""
        if target_pos is None: 
            return False
        
        if (int(agent_pos[0]), int(agent_pos[1])) == (int(target_pos[0]), int(target_pos[1])):
            return True

        path = self.finder.get_path(agent_pos, agent_dir, target_pos)
        if path:
            self.plan.extend(path)
            if actions:
                if isinstance(actions, list): 
                    self.plan.extend(actions)
                else: 
                    self.plan.append(actions)
            return True
        if actions:
            if isinstance(actions, list): 
                self.plan.extend(actions)
            else: 
                self.plan.append(actions)
            return True
        return False

    def is_facing(self, agent_pos, agent_dir, target_pos):
        """Check if agent is facing the target."""
        if target_pos is None: 
            return False
        dy, dx = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(agent_dir)]
        front_y, front_x = int(agent_pos[0]) + dy, int(agent_pos[1]) + dx
        return (front_y, front_x) == (target_pos[0], target_pos[1])


class SwapTaskPolicy(BaseExpertPolicy):
    """Policy for key and ball delivery tasks: put primary object in box and swap with secondary object."""
    def __init__(self, mode="key_delivery"):
        super().__init__()
        if mode == "key_delivery":
            self.p_id = ID_KEY
            self.s_id = ID_BALL
        elif mode == "ball_delivery":
            self.p_id = ID_BALL
            self.s_id = ID_KEY
            
        self.stage = "FETCH_PRIMARY"
        self.last_box_pos = None

    def get_action(self, obs_tensor, carrying_info):
        grid, agent_pos, agent_dir = self.init_grid_info(obs_tensor)
        
        p_pos = get_coords(self.obj_layer, self.p_id)
        s_pos = get_coords(self.obj_layer, self.s_id)
        box_pos = get_coords(self.obj_layer, ID_BOX)
        goal_pos = get_coords(self.obj_layer, ID_GOAL)
        if box_pos is not None:
            self.last_box_pos = box_pos

        if self.plan: return self.plan.pop(0)

        if self.stage == "FETCH_PRIMARY":
            self.move_to_target(agent_pos, agent_dir, p_pos)
            self.stage = "MATCH_COLOR_P"
            return self.get_action(obs_tensor, carrying_info)

        if self.stage == "MATCH_COLOR_P":
            if p_pos is None: 
                self.stage = "PICKUP_P"
                return self.get_action(obs_tensor, carrying_info)
            
            p_col = self.col_layer[p_pos[0], p_pos[1]]
            box_col = self.col_layer[box_pos[0], box_pos[1]] if box_pos is not None else -1
            
            if p_col != box_col and box_pos is not None:
                return ACT_TOGGLE
            else:
                self.stage = "PICKUP_P"
                return ACT_PICKUP

        if self.stage == "PICKUP_P":
            self.move_to_target(agent_pos, agent_dir, box_pos)
            self.stage = "PUT_IN_BOX"
            return self.get_action(obs_tensor, carrying_info)

        if self.stage == "PUT_IN_BOX":
            if carrying_info[1] == self.p_id:
                if box_pos is None:
                    box_pos = self.last_box_pos
                if box_pos is None:
                    return ACT_LEFT
                if self.is_facing(agent_pos, agent_dir, box_pos):
                    return ACT_TOGGLE
                else:
                    self.move_to_target(agent_pos, agent_dir, box_pos)
                    return self.get_action(obs_tensor, carrying_info)
            else:
                self.stage = "PICKUP_BOX"
                return self.get_action(obs_tensor, carrying_info)

        if self.stage == "PICKUP_BOX":
            if box_pos is None:
                box_pos = self.last_box_pos
            if box_pos is None:
                return ACT_LEFT
            if self.is_facing(agent_pos, agent_dir, box_pos):
                self.stage = "SWAP_SECONDARY"
                self.last_box_pos = box_pos
                return ACT_PICKUP
            else:
                self.move_to_target(agent_pos, agent_dir, box_pos)
                return self.get_action(obs_tensor, carrying_info)

        if self.stage == "SWAP_SECONDARY":
            if carrying_info[1] == self.s_id:
                self.stage = "DROP_NEAR_BOX"
                return self.get_action(obs_tensor, carrying_info)
            
            if s_pos is not None:
                self.move_to_target(agent_pos, agent_dir, s_pos, actions=ACT_DONE)
                self.stage = "DROP_NEAR_BOX"
                return self.get_action(obs_tensor, carrying_info)
            else:
                self.stage = "DROP_NEAR_BOX"
                return self.get_action(obs_tensor, carrying_info)

        if self.stage == "DROP_NEAR_BOX":
            if carrying_info[1] != self.s_id:
                self.stage = "MATCH_COLOR_S"
                return self.get_action(obs_tensor, carrying_info)
            
            ref_box_pos = box_pos if box_pos is not None else self.last_box_pos
            drop_pos = find_empty_position_near_box(self.obj_layer, self.col_layer, ref_box_pos, goal_pos, self.w, self.h)

            if drop_pos:
                path = self.finder.get_path(agent_pos, agent_dir, drop_pos)
                if path:
                    self.plan.extend(path)
                    self.plan.append(ACT_DROP)
                    self.stage = "MATCH_COLOR_S"
                    return self.get_action(obs_tensor, carrying_info)
                if abs(agent_pos[0]-drop_pos[0]) + abs(agent_pos[1]-drop_pos[1]) == 1:
                    target_dir = None
                    if agent_pos[0] < drop_pos[0]: target_dir = 1
                    elif agent_pos[0] > drop_pos[0]: target_dir = 3
                    elif agent_pos[1] < drop_pos[1]: target_dir = 0
                    else: target_dir = 2
                    if target_dir is not None and agent_dir != target_dir:
                        dir_diff = (target_dir - agent_dir) % 4
                        return ACT_RIGHT if dir_diff in [1,2] else ACT_LEFT
                    self.stage = "MATCH_COLOR_S"
                    return ACT_DROP

            dy, dx = [(1,0),(0,1),(-1,0),(0,-1)][agent_dir]
            front_y, front_x = agent_pos[0] + dy, agent_pos[1] + dx
            if 0 <= front_y < self.h and 0 <= front_x < self.w and self.obj_layer[front_y, front_x] == ID_EMPTY:
                self.stage = "MATCH_COLOR_S"
                return ACT_DROP
            return ACT_LEFT

        if self.stage == "MATCH_COLOR_S":
             s_pos = get_coords(self.obj_layer, self.s_id)
             if s_pos is None: 
                 self.stage = "GO_GOAL"
                 return self.get_action(obs_tensor, carrying_info)
             
             s_col = self.col_layer[s_pos[0], s_pos[1]]
             ref_box_pos = box_pos if box_pos is not None else self.last_box_pos
             box_col = self.col_layer[ref_box_pos[0], ref_box_pos[1]] if ref_box_pos is not None else (carrying_info[0] if carrying_info[1] == ID_BOX else 0)
             
             if s_col != box_col:
                 if self.is_facing(agent_pos, agent_dir, s_pos):
                     self.stage = "GO_GOAL"
                     return ACT_TOGGLE
                 else:
                     self.move_to_target(agent_pos, agent_dir, s_pos)
                     return self.get_action(obs_tensor, carrying_info)
             
             self.stage = "GO_GOAL"
             return self.get_action(obs_tensor, carrying_info)

        if self.stage == "GO_GOAL":
            self.move_to_target(agent_pos, agent_dir, goal_pos, actions=ACT_FORWARD)
            return self.get_action(obs_tensor, carrying_info)

        return ACT_FORWARD


class PlaceBothNearBoxPolicy(BaseExpertPolicy):
    """Policy for object mathching."""
    def __init__(self):
        super().__init__()
        self.objects_queue = [ID_KEY, ID_BALL]
        random.shuffle(self.objects_queue)
        
        self.check_pre_pickup = {ID_KEY: random.choice([True, False]), ID_BALL: random.choice([True, False])}
        
        self.current_obj = None
        self.stage = "NEXT_OBJ"

    def get_action(self, obs_tensor, carrying_info):
        grid, agent_pos, agent_dir = self.init_grid_info(obs_tensor)
        
        box_pos = get_coords(self.obj_layer, ID_BOX)
        goal_pos = get_coords(self.obj_layer, ID_GOAL)
        
        if self.plan: 
            return self.plan.pop(0)

        if self.stage == "NEXT_OBJ":
            if not self.objects_queue:
                self.stage = "GO_GOAL"
                return self.get_action(obs_tensor, carrying_info)
            self.current_obj = self.objects_queue.pop(0)
            self.stage = "FETCH"
            return self.get_action(obs_tensor, carrying_info)

        if self.stage == "FETCH":
            obj_pos = get_coords(self.obj_layer, self.current_obj)
            if obj_pos is None:
                self.stage = "NEXT_OBJ" 
                return self.get_action(obs_tensor, carrying_info)
            
            self.move_to_target(agent_pos, agent_dir, obj_pos)
            
            if self.check_pre_pickup[self.current_obj]:
                self.stage = "CHECK_COLOR_PRE"
            else:
                self.stage = "PICKUP"
            return self.get_action(obs_tensor, carrying_info)

        if self.stage == "CHECK_COLOR_PRE":
            obj_pos = get_coords(self.obj_layer, self.current_obj)
            if obj_pos is not None and self.is_facing(agent_pos, agent_dir, obj_pos):
                obj_col = self.col_layer[obj_pos[0], obj_pos[1]]
                box_col = self.col_layer[box_pos[0], box_pos[1]] if box_pos is not None else -1
                if obj_col != box_col:
                    return ACT_TOGGLE 
            
            self.stage = "PICKUP"
            return self.get_action(obs_tensor, carrying_info)

        if self.stage == "PICKUP":
            self.stage = "DELIVER"
            return ACT_PICKUP

        if self.stage == "DELIVER":
            if box_pos is None:
                self.stage = "NEXT_OBJ"
                return self.get_action(obs_tensor, carrying_info)
                
            drop_pos = find_empty_position_near_box(self.obj_layer, self.col_layer, box_pos, goal_pos, self.w, self.h)
            
            if drop_pos:
                path = self.finder.get_path(agent_pos, agent_dir, drop_pos)
                if path:
                    self.plan.extend(path)
                    self.plan.append(ACT_DROP)
                    
                    if not self.check_pre_pickup[self.current_obj]:
                        self.stage = "CHECK_COLOR_POST"
                    else:
                        self.stage = "NEXT_OBJ"
                    return self.get_action(obs_tensor, carrying_info)
            
            return ACT_LEFT

        if self.stage == "CHECK_COLOR_POST":
            front_y, front_x = agent_pos[0] + [(1,0),(0,1),(-1,0),(0,-1)][agent_dir][0], agent_pos[1] + [(1,0),(0,1),(-1,0),(0,-1)][agent_dir][1]
            obj_id = self.obj_layer[front_y, front_x]
            
            if obj_id == self.current_obj:
                obj_col = self.col_layer[front_y, front_x]
                box_col = self.col_layer[box_pos[0], box_pos[1]] if box_pos is not None else -1
                if obj_col != box_col:
                    return ACT_TOGGLE
            
            self.stage = "NEXT_OBJ"
            return self.get_action(obs_tensor, carrying_info)

        if self.stage == "GO_GOAL":
            self.move_to_target(agent_pos, agent_dir, goal_pos, actions=ACT_FORWARD)
            return self.get_action(obs_tensor, carrying_info)
            
        return ACT_FORWARD


class MultiTaskPolicy:
    """Multi-task policy that randomly selects between different task policies."""
    def __init__(self):
        self.policy = None
        self.reset()
    
    def reset(self):
        task_id = random.choices(['key_delivery', 'ball_delivery', 'object_matching'], weights=[1, 1, 1])[0]
        
        if task_id == 'key_delivery':
            self.policy = SwapTaskPolicy(mode="key_delivery")
        elif task_id == 'ball_delivery':
            self.policy = SwapTaskPolicy(mode="ball_delivery")
        elif task_id == 'object_matching':
            self.policy = PlaceBothNearBoxPolicy()
            

    def get_action(self, obs_tensor, carrying_info):
        return self.policy.get_action(obs_tensor, carrying_info)


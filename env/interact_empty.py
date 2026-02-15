from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from .interact_grid import *
import numpy as np

class EmptyEnv(InteractiveMiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
    
    def add_object(self, kind=None, color=None):
        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box'])

        color = np.random.choice(['red', 'blue'])
        
        if color == None:
            color = self._rand_color()
        
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = IKey(color)
        elif kind == 'ball':
            obj = IBall(color)
        elif kind == 'box':
            obj = IBox(color)

        return self.place_obj(obj)

class EmptyInteractEnv6x6_2_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        for _ in range(2):
            self.add_object()

register(
    id='MiniGrid-Empty-Interact-6x6-o2-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_2_object'
)

class EmptyInteractEnv6x6_3_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        self.add_object('box')
        self.add_object('ball')
        self.add_object('key')

register(
    id='MiniGrid-Empty-Interact-6x6-o3-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_3_object'
)

class EmptyInteractEnv6x6_3_object_train(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        self.add_object('box')
        self.add_object('ball', color='red')
        self.add_object('key', color='blue')

register(
    id='MiniGrid-Empty-Interact-6x6-o3-train-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_3_object_train'
)

class EmptyInteractEnv6x6_3_object_test(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        
        self.add_object('box')
        self.add_object('ball', color='blue')
        self.add_object('key', color='red')

register(
    id='MiniGrid-Empty-Interact-6x6-o3-test-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_3_object_test'
)


class EmptyInteractEnv6x6_4_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        for _ in range(4):
            self.add_object()

register(
    id='MiniGrid-Empty-Interact-6x6-o4-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_4_object'
)

class EmptyInteractEnv6x6_6_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        for _ in range(6):
            self.add_object()

register(
    id='MiniGrid-Empty-Interact-6x6-o6-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_6_object'
)

class EmptyInteractEnv6x6_8_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for _ in range(8):
            self.add_object()
            
register(
    id='MiniGrid-Empty-Interact-6x6-o8-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_8_object'
)

class EmptyInteractEnv6x6_10_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for _ in range(10):
            self.add_object()
            
register(
    id='MiniGrid-Empty-Interact-6x6-o10-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_10_object'
)

class EmptyInteractEnv6x6_12_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for _ in range(12):
            self.add_object()
            
register(
    id='MiniGrid-Empty-Interact-6x6-o12-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_12_object'
)

class EmptyInteractEnv6x6_14_object(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for _ in range(14):
            self.add_object()
            
register(
    id='MiniGrid-Empty-Interact-6x6-o14-v0',
    entry_point='gym_minigrid.envs:EmptyInteractEnv6x6_14_object'
)
import numpy as np
#import seaborn as sns

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity 这个entity是agent的最基本的物理类，和其他的受力的函数一起，这个是最基础的受力部分
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals 这个是用来判断agent的communication要不要加入noise用的
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # results 存储agent的运行结果
        self.result = []
        
        


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0 # 0维联络维数这个是MADDPG里面一个agent要得到其他agent的相关信息的信息的维数
        # position dimensionality 位置的维数
        self.dim_p = 2
        # color dimensionality 颜色的维数
        self.dim_color = 3
        # simulation timestep 0.1的timestep
        self.dt = 0.1
        # physical damping 物理衰减
        self.damping = 0.25
        # contact response parameters 回弹系数？
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default) 躲藏用的
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # 这个是用来存agents位置的
        self.agents_pos = []
        # 这个是记录agents、adversaries数量的
        self.num_good_agents = 0
        self.num_adversaries = 0

    # return all entities in the world
    @property # 这个property装饰之后这个函数调用就和调用属性一样，调用该函数不需要加()
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure 初始化距离的数据结构
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities 计算所有agents/adversaries 之间的距离，防止碰撞
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):  # hasattr判断agents[0]是否有"dummy"属性， 蠢货(dummy)模式？
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):  # 敌人(adversary)模式
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        dummy_colors = [(0, 0, 0)] * n_dummies
        adv_colors = sns.color_palette("OrRd_d", n_adversaries)
        good_colors = sns.color_palette("GnBu_d", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # update state of the world 这里面集合了好多动力学和运动学建模，这个world的建模感觉更像是碰碰车的动力学和运动学建模，有碰撞和受力的合力
    def step(self):
        # set actions for scripted agents agents的行为
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities 集中受力
        p_force = [None] * len(self.entities)
        # apply agent physical controls 物理受力
        p_force = self.apply_action_force(p_force)
        # apply environment forces 环境影响
        p_force = self.apply_environment_force(p_force)
        # integrate physical state 合力计算
        self.integrate_state(p_force)
        # update agent state 更新位置
        for agent in self.agents:
            self.update_agent_state(agent)
        # calculate and store distances between all entities 保存每一个agent的位姿
        if self.cache_dists:
            self.calculate_distances()


    # gather agent action forces 施加力，这里的受力用二维变量表示，满足二维受力的方法
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0  # 加入噪声
                p_force[i] = (agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise
        return p_force

    # gather physical forces acting on entities 这里设置的是碰撞的受力和撞墙的受力，这两部分统称环境受力
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # 还计算了衰减率
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt # 积分计算速度
            if entity.max_speed is not None: # 如果超过最大速度还会进行限制
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt  # 积分累加计算从而更新位置

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise # 这个是agent用来和其他agent之间通信用的部分，这里只是agent的action，在MADDPG算法里面，实际上还包含了除了agent之外的其他的信息，这些其他的信息由每一个实际脚本(scenarios)的相关observation决定    

    # get collision forces for any contact between two entities 两两之间的碰撞
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None] # neither entity moves
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall 这里计算的是entity撞到墙之后反弹回来的模拟
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
            ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force

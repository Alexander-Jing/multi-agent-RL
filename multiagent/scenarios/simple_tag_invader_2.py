import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 1
        world.num_adversaries = num_adversaries
        world.num_good_agents = num_good_agents
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.ghost = False
            agent.movable = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.05 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.07, 0.23]) if not agent.adversary else np.array([0.25, 0.4, 0.87])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.4, 0.87])
        # set random initial states
        for agent in world.agents:
            # 设定下agent 和 adversary的初始位置
            if agent.adversary:
                agent.state.p_pos = np.random.uniform(0.1, 0.8, world.dim_p) #np.random.uniform(-1.0, 1.0, world.dim_p)#
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.random.uniform(-0.8, -0.1, world.dim_p) #np.random.uniform(-1.0, 1.0, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.array([0.8, 0.8])#np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_results = ''
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)[0]  # 注意agent_reward修改过输出
        if not agent.adversary:
            main_results = self.agent_reward(agent, world)[1] 
        # 由于修改输出，所以上式取1
        return main_reward,main_results  # 注意其他文件里面和这个有关的也要改

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries  agent如果被抓则惩罚
        rew = 0
        collide_result = ''
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary) agent离开adversary越远越好
            for adv in adversaries:
                #print(agent.name)
                if agent.name=="agent 1":  # 两个进攻方，一个负责撞击防守方，一个负责攻击目标（远离拦截弹）
                    rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                else:
                    rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
            # we add the distance between agent and the landmarks as goals as reward negatively 我们加入由agent和landmarks之间距离构成的奖励用来训练
            for entity in world.landmarks:
                if not entity.boundary:
                    rew -= 0.1*np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
                    
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    collide_result = "collide adversary agent"
                    rew -= 10
            for entity in world.landmarks:
                if self.is_collision(entity,agent):
                    collide_result = "collide landmark agent"
                    rew += 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        # 这里修改下返回值，collide_result用来确定此次攻防的结果
        return rew,collide_result

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        #print(agent.name)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents) 对于adversary的reward：和agent的距离，越小越好（注意是累加的）
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:  # 如果撞在一起的话，reward再加一部分
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    # 这个是MADDPG里面对于每一个agent/adversary的observation观测空间
    # 每一个agent/adversary的observation这一部分包括：
    #  agent/adversary和landmark之间的相对坐标距离 entity_pos
    #  其他agents/adversaries 的行为action (MADDPG论文里面标准的一部分) comm
    #  其他agents/adversaries 的位置和这一个agent/adversary的位置的相对坐标距离 other_pos，计算方式和上面那个entity_pos一样
    #  其他agents(注意没有adversaries，这里只有agents) 的速度
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)  # 观测空间observation是agent和landmark之间的相对坐标差值（坐标之差）
        # communication of all other agents 这一部分是观测空间里面包含的其他agents的信息
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        #print(np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

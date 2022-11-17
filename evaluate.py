import argparse
from multiagent.core import World
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    success_done = 0  # 这个是统计有没有成功突防的

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env._reset()
        if config.save_gifs:
            frames = []
            #print(env._render('rgb_array'))
            frames.append(env._render('rgb_array',False)[0])
        #env._render('human')
        
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env._step(actions)
            
            if config.save_gifs:
                frames.append(env._render('rgb_array',False)[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            #env._render('human')
        # 判断下是否成功突防，注意results第一个值是"collide landmark agent"即成功突防
        #print(env.results)        
        if  next(sub for sub in env.results if sub) == "collide landmark agent" :
            success_done += 1.0
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
        # 显示出物理位置的变化 env.pos_log是其坐标，
        #print(np.array(env.pos_log).shape)
    print("success rate: {:4.2f} % ".format(success_done/config.n_episodes * 100))  # 计算突防率
    env.close()

def run_times(config):

    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    success_done_all = []
    failure_done_all = []
    for test_i in range(config.test_times):
        success_done = 0  # 这个是统计有成功突防的
        failure_done = 0  # 这个是统计没有成功突防的
        print("run_times %i of %i" % (test_i+1, config.test_times))
        
        for ep_i in range(config.n_episodes):
            print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
            obs = env._reset()
            
            if config.save_gifs:
                frames = []
                #print(env._render('rgb_array'))
                frames.append(env._render('rgb_array',False)[0])
            #env._render('human')
            
            for t_i in range(config.episode_length):
                calc_start = time.time()
                # rearrange observations to be per agent, and convert to torch Variable
                torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                    requires_grad=False)
                            for i in range(maddpg.nagents)]
                # get actions as torch Variables
                torch_actions = maddpg.step(torch_obs, explore=False)
                # convert actions to numpy arrays
                actions = [ac.data.numpy().flatten() for ac in torch_actions]
                obs, rewards, dones, infos = env._step(actions)
                
                if config.save_gifs:
                    frames.append(env._render('rgb_array',False)[0])
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                #env._render('human')
            # 判断下是否成功突防，注意results第一个值是"collide landmark agent"即成功突防
            #print(env.results) 
            success_done_add,failure_done_add = 0,0
            for result in env.results:   
                if [x for x in result if x != ''] != []:
                    if  next(sub for sub in result if sub) == "collide landmark agent" :
                        success_done_add += 1.0  # 这里仅仅是统计进攻方只有1个的情况下的情况
                    else :
                        failure_done_add += 1.0  # 这里仅仅是统计进攻方只有1个的情况下的情况
            # 从相加部分判断实际的这一次的突防状况
            if success_done_add >= 1:
                success_done += 1
            if failure_done_add == len([result for result in env.results if result!=[]]):
                failure_done += 1
            # 显示轨迹的
            if config.save_PNGs:
                sns.set_style("darkgrid")    
                ax = plt.gca()
                ax.set_aspect(1)
                plt.xlim((-1.0,1.0))
                plt.ylim((-1.0,1.0))
                plt.scatter(0.8,0.8,color='b',s=3000)
                #print(np.array(env.pos_log).shape)
                for i,pos in enumerate(env.pos_log):
                    if i < env.world.num_adversaries: # 一直是先敌人的数据
                        defender_x = np.array(pos)[:,0]
                        defender_y = np.array(pos)[:,1]
                        plt.plot(defender_x,defender_y,color='b')
                    else:
                        invader_x = np.array(pos)[:,0]
                        invader_y = np.array(pos)[:,1]
                        plt.plot(invader_x,invader_y,color='r')
                plt.savefig('MADDPG-test'+'ep_i '+str(ep_i)+' test_i '+str(test_i))
                
                plt.show()
                plt.close()
                print('save successfully')    
                
            if config.save_gifs:
                gif_num = 0
                while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                    gif_num += 1
                imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                                frames, duration=ifi)
        success_done_all.append(success_done/config.n_episodes)
        failure_done_all.append(failure_done/config.n_episodes)
        print("success rate: {:4.2f} % ".format(success_done/config.n_episodes * 100))  # 计算突防率
        env.close()
    print("invader : test best rate: {:4.2f} % , test average rate: {:4.2f} %".\
                  format(max(success_done_all)*100,sum(success_done_all)/len(success_done_all)*100))  # 计算并显示突防率的最大值和平均值
    print("defender : test best rate: {:4.2f} % , test average rate: {:4.2f} %".\
                  format(max(failure_done_all)*100,sum(failure_done_all)/len(failure_done_all)*100))  # 计算并显示突防率的最大值和平均值


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment",default="simple_tag_invader_2")
    parser.add_argument("model_name",
                        help="Name of model",default="Predator-prey_2")
    parser.add_argument("run_num", default=4, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")#,default="--save_gifs")
    parser.add_argument("--save_PNGs", action="store_true",
                        help="Saves PNG of the trail of each episode into model directory")#,default="--save_PNGs")  # 这个是新加进来的，用于存储每一个场次训练的轨迹
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=35, type=int)
    parser.add_argument("--test_times", default=1, type=int)  # 这是新加进来的用来做多次测试区平均突防率的测试数量
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args() 

    #run(config)
    run_times(config)
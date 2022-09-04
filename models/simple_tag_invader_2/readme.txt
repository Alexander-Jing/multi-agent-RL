这里是二打一的基于欺骗-偷袭战术设置的
simple_tag_invader_2
run3 二打一 训练 欺骗-偷袭战术 两边MADDPG，全局随机位置，进攻方加速度速度上限 6.0，防守方3.0 噪声高斯 初始scale 0.3
测试 固定位置(0.1,0.8) 
invader : test best rate: 87.00 % , test average rate: 83.20 %
defender : test best rate: 23.00 % , test average rate: 16.80 %
进攻方加速度速度上限 4.0，防守方3.0 噪声高斯
invader : test best rate: 91.00 % , test average rate: 87.20 %
defender : test best rate: 18.00 % , test average rate: 12.80 %

run4 二打一 训练 欺骗-偷袭战术 两边MADDPG，全局随机位置，进攻方加速度速度上限 4.0，防守方3.0 噪声高斯 初始scale 0.3
测试 固定位置(0.1,0.8) 
invader : test best rate: 77.00 % , test average rate: 75.00 %
defender : test best rate: 27.00 % , test average rate: 25.00 %

run5 二打一 训练 欺骗-偷袭战术 两边MADDPG，全局随机位置，进攻方加速度速度上限 4.0，防守方3.0 噪声高斯 初始scale 1.0
测试 固定位置(0.1,0.8) 
invader : test best rate: 98.00 % , test average rate: 97.20 %
defender : test best rate: 4.00 % , test average rate: 2.80 %
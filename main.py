import gym
import pyglet
import matplotlib.pyplot as plt
import numpy as np
import random
plt.style.use('_mpl-gallery')

import torch
import funct_and_nn as DQL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# loads envirorment
env = gym.make('CartPole-v1')
#env = gym.make("LunarLander-v2")

number_of_episode = 5000
max_time = 1000

resulting_time_ls = []

# set q earning values
alpha = .5
gamma = .5
start_ep = 1
end_ep = 0
ep_states = 1000

# init networks
policy_net = DQL.NeuralNetwork(4,100,2).to(device)
target_net = DQL.NeuralNetwork(4,100,2).to(device)

read_filename = ""

if(read_filename != ""):
    start_ep = 0
    policy_net.load_state_dict(torch.load(read_filename))
    target_net.load_state_dict(torch.load(read_filename))
    print("Loaded")

# true if you only want the policy to run and no learning to occur
is_only_policy = False


# set net varibles
learning_rate = .0005
#learning_rate = .00005

# set loss funct
loss_fn = torch.nn.MSELoss()

# set optimizers
#optimizer_policy_net = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
#optimizer_target_net = torch.optim.Adam(target_net.parameters(), lr=learning_rate)
optimizer_policy_net = torch.optim.SGD(policy_net.parameters(), lr=learning_rate, momentum=0.9)
optimizer_target_net = torch.optim.SGD(target_net.parameters(), lr=learning_rate, momentum=0.9)
#optimizer_policy_net = torch.optim.SGD(policy_net.parameters(), lr=learning_rate)

#scheduler_policy = torch.optim.lr_scheduler.StepLR(optimizer_policy_net, step_size=250, gamma=.1)
scheduler_policy = torch.optim.lr_scheduler.CyclicLR(optimizer_policy_net, base_lr=0.0000005, max_lr=0.0005, mode="triangular2")
scheduler_target = torch.optim.lr_scheduler.CyclicLR(optimizer_target_net, base_lr=0.0000005, max_lr=0.0005, mode="triangular2")
#scheduler_policy = torch.optim.lr_scheduler.CyclicLR(optimizer_policy_net, base_lr=0.000005, max_lr=0.01, mode="triangular2")
# init memory
capacity = 10000
batch_size = 50
memory = DQL.Memory(capacity)
memory_count = 0
memory_refresh_rate = 10 # set net equalization net counter

# sets total_state_count
ep_state_count = 0

# temp
# define action list
action_ls = [0, 1]


checkpoint_counter = 0

checkpoint_width = 400
start_check_point = 100

current_checkpoint_score = 0
max_checkpoint_score = 0

print_counter = 0


for i_episode in range(number_of_episode):


    if(checkpoint_counter > checkpoint_width and i_episode > start_check_point):

        # calc new current score
        for i in range(checkpoint_width):
            current_checkpoint_score += int(resulting_time_ls[-i])
        current_checkpoint_score = current_checkpoint_score/checkpoint_width


        if (current_checkpoint_score > max_checkpoint_score):
            max_checkpoint_score = current_checkpoint_score
            torch.save(policy_net.state_dict(), "checkpoint_weights.pth")  # save current check point
        else:
            print("RESET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            policy_net.load_state_dict(torch.load("checkpoint_weights.pth"))
            target_net.load_state_dict(torch.load("checkpoint_weights.pth"))
            ep_state_count = 0

        checkpoint_counter = 0

    else:
        checkpoint_counter += 1

    total_reward = 0

    observation = env.reset()

    for t in range(max_time):   # game loop

        #env.render()  # loads game window not needed

        # get pre state
        pre_state = observation

        # calc epsilon
        epsilon = DQL.Epsilon_Decay(start_ep, end_ep, ep_states, ep_state_count)

        # policy # set actions
        action = DQL.policy(pre_state, action_ls, policy_net, epsilon)
        #action = env.action_space.sample()

        # update envirorment
        observation, reward, done, info = env.step(action)

        # get post state
        post_state = observation

        total_reward += reward

        if done:
            if(print_counter == 4):

                print("Episode finished after {} timesteps".format(t + 1))
                for param_group in optimizer_policy_net.param_groups:
                    print("episode: ", i_episode, " epsilon: ", epsilon, " learning rate: ", param_group['lr'] * 10000)
                print_counter = 0

            else:
                print_counter += 1

            resulting_time_ls.append(format(t + 1))
            break

        #if(not is_only_policy):

        # construct exprence and store in memory
        memory.push(DQL.Exp(pre_state, action, post_state, (total_reward ** .5), done))
        #memory.push(DQL.Exp(pre_state, action, post_state, total_reward, done))

        # Update policy net
        if (memory.is_sample_availble(batch_size)):
            # sample memory
            sample = memory.sample(batch_size)

            # Update net
            # DQL.Update_policy_net(sample, gamma, loss_fn, optimizer_policy_net, policy_net, target_net)
            # DQL.Update_policy_net(sample, gamma, loss_fn, optimizer_policy_net, scheduler_policy, policy_net, target_net)
            DQL.Update_policy_net_double_Q_learning(sample, gamma, loss_fn, optimizer_policy_net, scheduler_policy, policy_net, target_net)
            # DQL.Update_policy_net_double_Q_learning(sample, gamma, loss_fn, optimizer_policy_net, policy_net, target_net)

        # sets target net equal to policy net
        if (memory_count == memory_refresh_rate):
            memory_count = 0
            target_net.load_state_dict(policy_net.state_dict())


        ep_state_count += 1


env.close()




# data analysis no DQN code past this point




# plot
fig, ax = plt.subplots()
x = []
y = []
sum = 0
max_y = 0

moving_avg = []
moving_avg_width = 20

for i in range(len(resulting_time_ls)):
    x.append(i)
    y.append(int(resulting_time_ls[i]))

    sum += int(resulting_time_ls[i])

    if(int(resulting_time_ls[i]) > max_y):
        max_y = int(resulting_time_ls[i])

    if(i > moving_avg_width):
        moving_avg_sum = 0
        for n in range(moving_avg_width):
            moving_avg_sum += int(resulting_time_ls[i - n])
        moving_avg.append(moving_avg_sum/moving_avg_width)
    else:
        moving_avg.append(0)


avg = sum/len(resulting_time_ls)


ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
ax.plot(x, moving_avg, linewidth=2.0, color = 'g')
ax.set(xlim=(0, number_of_episode), xticks=np.arange(1, number_of_episode), ylim=(0, (max_y + 5)), yticks=np.arange(1, (max_y + 5)))


#print(resulting_time_ls)
print("avg: ",avg)

plt.show()



is_save = input("Do you want to save this policy net? (Y/N)")
if(is_save == "y" or is_save == "Y"):
    torch.save(policy_net.state_dict(), 'model_weights.pth')


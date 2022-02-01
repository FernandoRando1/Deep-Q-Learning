import torch
from torch import nn
import torch.nn.functional as F
import random

import numpy

class Exp:
    def __init__(self,state_,action_,next_state_,reward_,is_terminal_):
        self.state = state_
        self.action = action_
        self.next_state = next_state_
        self.reward = reward_
        self.is_treminal = is_terminal_


class Memory:
    def __init__(self, capacity_):
        self.capacity = capacity_
        self.memory = []

    def push(self,exp_):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(exp_)

    def sample(self, batch_size_):
        return random.sample(self.memory, batch_size_)

    def is_sample_availble(self, batch_size_):
        return len(self.memory) >= batch_size_


class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.hidden3 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.hidden4 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)             # linear output
        return x

"""
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = nn.Linear(n_hidden, int(n_hidden/2))  # hidden layer
        self.hidden3 = nn.Linear(int(n_hidden/2), int(n_hidden/4))  # hidden layer
        self.hidden4 = nn.Linear(int(n_hidden/4), int(n_hidden/8))  # hidden layer
        self.predict = nn.Linear(int(n_hidden/8), n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)             # linear output
        return x
"""

def Epsilon_Decay(starting_ep_,end_ep_,over_states_,states_taken_):
    # basic liner decay
    # over_states_ = how many state to reach end for start
    # states_taken_ = current state number or number of state that have occured
    if(states_taken_ > over_states_):
        return 0
    ep = starting_ep_ + (((end_ep_ - starting_ep_) / over_states_ ) * states_taken_ )
    #print(ep)
    return ep


def policy(state_, action_ls_, policy_net_, epsilon_):

    if(random.random() > epsilon_):
        # policy_net_out = values form policy nn
        # with no_grad()
        with torch.no_grad():
            pred = policy_net_(torch.FloatTensor(state_))
            #print("Q_policy_net: out: ", pred)
            softmax = nn.Softmax(dim=0)
            pred_probab = softmax(pred)
            act_index = pred_probab.argmax(0)
            #print(pred,action_ls_[act_index])
            return action_ls_[act_index]
    else:
        return action_ls_[random.randint(0,len(action_ls_) - 1)]

    # action_ls_ = list of actions that corrispond to the nn
    # returns action


def Update_policy_net(sample_ls_, gamma_, loss_funct_, opt_funct_, shed_funct_, policy_net_, target_net_):
#def Update_policy_net(sample_ls_, gamma_, loss_funct_, opt_funct_, policy_net_, target_net_):
    # make sample experience tensor
    pre_state_ls = []
    post_state_ls = []
    action_ls = []
    reward_ls = []
    is_treminal_ls = []

    for exp in sample_ls_:
        pre_state_ls.append(exp.state)
        post_state_ls.append(exp.next_state)
        action_ls.append(exp.action)
        reward_ls.append(exp.reward)
        is_treminal_ls.append(exp.is_treminal)

    #print(len(pre_state_ls))

    pre_state_tensor = torch.reshape(torch.FloatTensor(numpy.array(pre_state_ls)),[len(pre_state_ls), 4])
    post_state_tensor = torch.reshape(torch.FloatTensor(numpy.array(post_state_ls)), [len(post_state_ls), 4])
    action_tensor = torch.reshape(torch.LongTensor(numpy.array(action_ls)), [len(action_ls), 1])
    reward_tensor = torch.reshape(torch.FloatTensor(numpy.array(reward_ls)), [len(reward_ls), 1])
    is_terminal_tensor = torch.reshape(torch.BoolTensor(numpy.array(is_treminal_ls)), [len(is_treminal_ls), 1])


    # calc Q_target table
    with torch.no_grad():
        Q_target_table = target_net_(post_state_tensor)

    # calc Q_target
    Q_target_max = torch.reshape(torch.amax(Q_target_table,1),[Q_target_table.shape[0],1])

    # calc y values
    bellman_tensor = torch.add(reward_tensor, Q_target_max * gamma_)
    y_values_tensor = torch.where(is_terminal_tensor,reward_tensor,bellman_tensor)

    # calc Q_table_values (for all actions)
    Q_policy_table = policy_net_(pre_state_tensor)

    Q_policy_values_tensor = Q_policy_table.gather(1, action_tensor.view(-1,1)).view(-1)
    Q_policy_values_tensor = torch.reshape(Q_policy_values_tensor,[Q_policy_values_tensor.shape[0],1])

    # calc loss and back propigate
    loss = loss_funct_(Q_policy_values_tensor, y_values_tensor)

    # Backpropagation
    opt_funct_.zero_grad()
    loss.backward()
    opt_funct_.step()
    #shed_funct_.step()


def Update_policy_net_double_Q_learning(sample_ls_, gamma_, loss_funct_, opt_funct_, shed_funct_, policy_net_, target_net_):
#def Update_policy_net_double_Q_learning(sample_ls_, gamma_, loss_funct_, opt_funct_, policy_net_, target_net_):
    # make sample experience tensor
    pre_state_ls = []
    post_state_ls = []
    action_ls = []
    reward_ls = []
    is_treminal_ls = []

    for exp in sample_ls_:
        pre_state_ls.append(exp.state)
        post_state_ls.append(exp.next_state)
        action_ls.append(exp.action)
        reward_ls.append(exp.reward)
        is_treminal_ls.append(exp.is_treminal)

    #print(len(pre_state_ls))

    pre_state_tensor = torch.reshape(torch.FloatTensor(numpy.array(pre_state_ls)),[len(pre_state_ls), 4])
    post_state_tensor = torch.reshape(torch.FloatTensor(numpy.array(post_state_ls)), [len(post_state_ls), 4])
    action_tensor = torch.reshape(torch.LongTensor(numpy.array(action_ls)), [len(action_ls), 1])
    reward_tensor = torch.reshape(torch.FloatTensor(numpy.array(reward_ls)), [len(reward_ls), 1])
    is_terminal_tensor = torch.reshape(torch.BoolTensor(numpy.array(is_treminal_ls)), [len(is_treminal_ls), 1])

    # calc Q_target choice
    with torch.no_grad():
        Q_target_table_choice = torch.reshape(target_net_(post_state_tensor).argmax(1),[post_state_tensor.shape[0],1])
        #print(Q_target_table[0])
        #print(Q_target_table.argmax(1)[0])

    # calc Q_policy_value(Q_target choice)
    with torch.no_grad():
        Q_policy_target_table = policy_net_(post_state_tensor)
    Q_policy_target_value = Q_policy_target_table.gather(1, Q_target_table_choice.view(-1,1)).view(-1)
    Q_policy_target_value = torch.reshape(Q_policy_target_value,[Q_policy_target_value.shape[0],1])


    # calc Q_table_values (for all actions)
    Q_policy_table = policy_net_(pre_state_tensor)

    Q_policy_values_tensor = Q_policy_table.gather(1, action_tensor.view(-1, 1)).view(-1)
    Q_policy_values_tensor = torch.reshape(Q_policy_values_tensor, [Q_policy_values_tensor.shape[0], 1])

    # calc y values
    bellman_tensor = torch.add(reward_tensor, Q_policy_target_value * gamma_)
    y_values_tensor = torch.where(is_terminal_tensor,reward_tensor,bellman_tensor)



    # calc loss and back propigate
    loss = loss_funct_(Q_policy_values_tensor, y_values_tensor)

    # Backpropagation
    opt_funct_.zero_grad()
    loss.backward()
    opt_funct_.step()
    shed_funct_.step()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import deque
import random
import numpy as np
from tqdm import tqdm  # 导入tqdm
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图

# 超参数设置
batch_size = 64
learning_rate = 0.01
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 贪婪策略的epsilon
target_update = 20  # 每多少次更新一次目标网络
num_episodes = 1 # 训练的轮数

# CIFAR-10 数据集的转换和加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# 定义 Q 网络（DQN）
class DQN(nn.Module):
    def __init__(self, input_channels=3, num_actions=10):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)  # 添加padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 添加padding
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 添加padding


        with torch.no_grad():
            self._test_input = torch.zeros(1, input_channels, 32, 32)  # 假设输入图片尺寸为 32x32
            self._flatten_size = self._get_conv_output(self._test_input)

        self.fc1 = nn.Linear(self._flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):

        x = self.conv1(shape)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建经验回放缓冲区（Replay Buffer）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# 选择动作
def select_action(state, policy_net, epsilon, device):
    if random.random() < epsilon:

        return [random.randrange(10) for _ in range(state.size(0))]
    else:

        with torch.no_grad():
            return policy_net(state.to(device)).argmax(dim=1).cpu().numpy().tolist()


# 优化模型
def optimize_model(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    if replay_buffer.size() < batch_size:
        return

    transitions = replay_buffer.sample(batch_size)
    batch = list(zip(*transitions))
    states, actions, rewards, next_states, dones = batch

    states = torch.cat(states).to(device)
    next_states = torch.cat(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    dones = torch.tensor(dones).to(device)

    state_action_values = policy_net(states).gather(1, actions.unsqueeze(1))
    next_state_values = target_net(next_states).max(1)[0].detach()
    expected_state_action_values = rewards + (gamma * next_state_values * (1 - dones))

    loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 计算验证集准确率
def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# 训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(10000)

# 记录训练损失和验证集准确率
train_losses = []
val_accuracies = []

# 训练过程
for episode in range(num_episodes):
    policy_net.train()
    running_loss = 0
    correct = 0
    total = 0

    # 使用 tqdm 显示训练进度条
    for i, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader),
                                     desc=f'Episode {episode + 1}/{num_episodes}'):
        inputs, targets = inputs.to(device), targets.to(device)

        state = inputs
        actions = select_action(state, policy_net, epsilon, device)

        # 模拟“奖励”机制，预测错误时扣分，预测正确时加分
        reward = [1 if a == t.item() else -1 for a, t in zip(actions, targets)]
        done = [1 if a == t.item() else 0 for a, t in zip(actions, targets)]

        # 存储状态转移
        for s, a, r, ns, d in zip(state, actions, reward, state, done):
            replay_buffer.push((s.unsqueeze(0), a, r, ns.unsqueeze(0), d))

        optimize_model(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)

        running_loss += sum(reward)
        correct += sum([1 for a, t in zip(actions, targets) if a == t.item()])
        total += len(actions)

    # 计算每个回合结束后的验证集准确度
    val_accuracy = evaluate(policy_net, testloader, device)
    train_losses.append(running_loss / total)
    val_accuracies.append(val_accuracy)
    print(
        f"Episode {episode + 1}/{num_episodes}, Loss: {running_loss / total:.4f}, Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    # 每个指定的回合数更新一次目标网络
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

# 测试模型
test_accuracy = evaluate(policy_net, testloader, device)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# 绘制损失和验证集准确率在同一张图片上
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制损失（左边的Y轴）
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Loss', color='blue')
ax1.plot(range(1, num_episodes + 1), train_losses, label='Training Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建另一个Y轴（右边的Y轴）用于绘制准确率
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy (%)', color='orange')
ax2.plot(range(1, num_episodes + 1), val_accuracies, label='Validation Accuracy', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 添加标题和网格
plt.title('Training Loss and Validation Accuracy per Episode')
plt.grid(True)
plt.tight_layout()

plt.show()

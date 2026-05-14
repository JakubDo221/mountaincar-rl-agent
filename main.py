import gymnasium as gym#RL
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

import torch#PyTorch – fr do sieci neuronowych.
import torch.nn as nn#warstw sieci (Linear, Conv)
import torch.nn.functional as F#funkcje aktywacji

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== ENV =====
env = gym.make("MountainCarContinuous-v0")
state, info = env.reset(seed=101)
np.random.seed(101)

print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)

# ===== AGENT =====
class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super().__init__()
        self.env = env
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.s_size, self.h_size)#stan → warstwa ukryta
        self.fc2 = nn.Linear(self.h_size, self.a_size)#warstwa ukryta → akcja

    def set_weights(self, weights):#Polaczenie miedzy stanem a fc1-fc2 a akcja
        s, h, a = self.s_size, self.h_size, self.a_size
        fc1_end = (s*h)+h

        self.fc1.weight.data = torch.tensor(
            weights[:s*h].reshape(h, s), dtype=torch.float32
        )
        self.fc1.bias.data = torch.tensor(
            weights[s*h:fc1_end], dtype=torch.float32
        )

        self.fc2.weight.data = torch.tensor(
            weights[fc1_end:fc1_end+(h*a)].reshape(a, h), dtype=torch.float32
        )
        self.fc2.bias.data = torch.tensor(
            weights[fc1_end+(h*a):], dtype=torch.float32#if h*a==0, wynik==bias(punkt, ustawiene startowe)
        )

    def get_weights_dim(self):#ile liczb (wag + biasów) potrzeba, żeby opisać całą sieć
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size

    def forward(self, x):#myslenie agenta automatczne wywolanie po: action = agent(state)
        x = F.relu(self.fc1(x))#ujemne wartości → 0, dodatnie → bez zmian
        x = torch.tanh(self.fc2(x))#duże wartości są ściskane do zakresu [-1, 1]
        return x#zwraca akcje agenta

    def evaluate(self, weights, gamma=1.0, max_t=300):#uruchamia agenta w środowisku i liczy, jak dobrze sobie poradził dla danych wag
        self.set_weights(weights)
        state, _ = self.env.reset()
        total_reward = 0.0#wynik

        for t in range(max_t):#petla czasu(max krokow)
            state_t = torch.tensor(state, dtype=torch.float32).to(device)

            with torch.no_grad():
                action = self(state_t).cpu().numpy()

            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward * (gamma ** t)#nagroda zdyskontowana, wcześniejsze nagrody → ważniejsze

            if terminated or truncated:#koniec jezeli cel albo brak czasu
                break

        return total_reward

agent = Agent(env).to(device)

# ===== CEM ===== Losuj wiele agentow → sprawdź ich → zostaw najlepszych → losuj nowe wokół nich
def cem(
    n_iterations=300,     # zmniejszone z 300
    max_t=1000,           # zmniejszone z 1000 maksymalna liczba krokow na 1 epizod
    gamma=1.0,            #nagrody
    pop_size=50,          # zmniejszone z 50 population size
    elite_frac=0.2,       #20% agentow przetrwa do nastepnej iter
    sigma=1.0,            #wagi
    print_every=1         # drukuj co iterację
):
    n_elite = int(pop_size * elite_frac)#number of elite
    scores_deque = deque(maxlen=100)#Przechowuje ostatnie 100 wyników
    scores = []#Lista wszystkich wyników

    best_weight = sigma * np.random.randn(agent.get_weights_dim())#VVVV Losuj wiele agentow → sprawdź ich → zostaw najlepszych → losuj nowe wokół nich

    for i in range(1, n_iterations + 1):
        weights_pop = [
            best_weight + sigma * np.random.randn(agent.get_weights_dim())#worzysz 50 (pop_size) wariantów age
            for _ in range(pop_size)
        ]

        rewards = np.array([
            agent.evaluate(w, gamma, max_t) for w in weights_pop
        ])

        elite_idxs = rewards.argsort()[-n_elite:]# sortowanie, wybieranie najlepszyzch
        elite_weights = [weights_pop[i] for i in elite_idxs]#selekcja
        best_weight = np.mean(elite_weights, axis=0)#nowy jest srednia najlepszych agentow

        reward = agent.evaluate(best_weight)
        scores.append(reward)
        scores_deque.append(reward)

        print(f"Iteration {i}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= 90.0:#Jeśli agent jest wystarczająco dobry → stop
            print(f"\nSolved in {i} iterations!")
            break

    agent.set_weights(best_weight)
    torch.save(agent.state_dict(), "checkpoint.pth")#Zapis najlepszego mózgu
    return scores

# ===== Uruchomienie =====
scores = cem()

# ===== PLOT =====
plt.plot(scores)
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.show()
# ===== LOAD WEIGHTS =====
agent.load_state_dict(torch.load('checkpoint.pth'))

# ===== VISUALIZACJA W COLABIE =====
from matplotlib import animation
from IPython.display import HTML

env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

state, _ = env.reset()
frames = []

for _ in range(500):
    state_t = torch.tensor(state, dtype=torch.float32).to(device)
    with torch.no_grad():
        action = agent(state_t).cpu().numpy()

    state, reward, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

env.close()

# ===== ANIMACJA =====
fig = plt.figure()
plt.axis("off")
im = plt.imshow(frames[0])

def animate(i):
    im.set_array(frames[i])
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=30)
HTML(anim.to_jshtml())
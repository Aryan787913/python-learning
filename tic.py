import numpy as np
import random
import tkinter as tk
from tkinter import scrolledtext
from scipy.cluster.hierarchy import linkage, fcluster

# ============================================================
# Q-LEARNING AGENT & CLUSTERING LOGIC
# ============================================================
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.state_cluster = {}
        self.cluster_Q = {}
        self.reward_list = []
        
        self.win_states = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]

    def available_actions(self, state):
        return [i for i, val in enumerate(state) if val == 0]

    def check_winner(self, state):
        for w in self.win_states:
            s = sum(state[i] for i in w)
            if s == 3: return 1, True    # Player 1 (X) wins
            if s == -3: return -1, True  # Player -1 (O/Agent) wins
        if 0 not in state:
            return 0, True               # Draw
        return 0, False

    def train_self_play(self, episodes=1000):
        """Simulates games against a random opponent to build the Q-table."""
        total_reward = 0
        for _ in range(episodes):
            state = tuple(np.zeros(9, dtype=int))
            while True:
                # Agent (1) Move
                actions = self.available_actions(state)
                if random.random() < self.epsilon:
                    action = random.choice(actions)
                else:
                    qvals = [self.Q.get((state, a), 0) for a in actions]
                    action = actions[np.argmax(qvals)]

                # Apply Agent move
                next_state_list = list(state)
                next_state_list[action] = 1
                next_state = tuple(next_state_list)
                reward, done = self.check_winner(next_state)

                if done:
                    self.update_q(state, action, reward, next_state)
                    total_reward += reward
                    break

                # Opponent (-1) Random Move
                opp_actions = self.available_actions(next_state)
                opp_action = random.choice(opp_actions)
                next_state_list[opp_action] = -1
                next_state = tuple(next_state_list)
                reward, done = self.check_winner(next_state)
                
                # Update Q after opponent move
                self.update_q(state, action, -reward, next_state)

                if done:
                    total_reward -= reward 
                    break

                state = next_state
                
        self.reward_list.append(f"Trained {episodes} eps | Net Reward: {total_reward}")
        self.update_clusters()

    def update_q(self, state, action, reward, next_state):
        next_actions = self.available_actions(next_state)
        max_q_next = max([self.Q.get((next_state, a), 0) for a in next_actions], default=0)
        current_q = self.Q.get((state, action), 0)
        self.Q[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)

    def update_clusters(self):
        """Extracts Q-vectors, removes zeros (Critical Fix), and builds Hierarchical Clusters."""
        if not self.Q: return

        states = list(set([s for s, _ in self.Q.keys()]))
        Q_vectors = np.array([[self.Q.get((s, a), 0) for a in range(9)] for s in states])

        valid_states = []
        valid_Q = []

        # CRITICAL FIX: Remove zero Q-vectors
        for i, q in enumerate(Q_vectors):
            if not np.allclose(q, 0):
                valid_states.append(states[i])
                valid_Q.append(q)

        if len(valid_Q) < 2: return # Not enough data to cluster

        valid_Q = np.array(valid_Q)
        
        # Hierarchical Clustering
        Z = linkage(valid_Q, method="average", metric="cosine")
        clusters = fcluster(Z, t=5, criterion="maxclust")

        # Map States -> Clusters
        self.state_cluster = {valid_states[i]: clusters[i] for i in range(len(valid_states))}

        # Build Cluster-level Q-table
        self.cluster_Q = {}
        for s, a in self.Q.keys():
            if s in self.state_cluster:
                c = self.state_cluster[s]
                self.cluster_Q[(c, a)] = self.cluster_Q.get((c, a), 0) + self.Q.get((s, a), 0)

    def choose_cluster_action(self, state):
        """Chooses an action based on the grouped Cluster Q-Table."""
        actions = self.available_actions(state)
        if not actions: return None
        
        # Fallback if state hasn't been clustered yet
        if state not in self.state_cluster:
            return random.choice(actions)
            
        c = self.state_cluster[state]
        qvals = [self.cluster_Q.get((c, a), 0) for a in actions]
        return actions[np.argmax(qvals)]


# ============================================================
# TKINTER GUI
# ============================================================
class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning Cluster Tic-Tac-Toe")
        
        self.agent = QLearningAgent()
        self.board = np.zeros(9, dtype=int)
        self.agent_history = []
        
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        # Top Frame: Controls
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack()
        
        train_btn = tk.Button(control_frame, text="Train Agent (1000 Games)", command=self.train_agent, bg="lightblue")
        train_btn.grid(row=0, column=0, padx=5)
        
        reset_btn = tk.Button(control_frame, text="Reset Board", command=self.reset_game)
        reset_btn.grid(row=0, column=1, padx=5)

        self.status_label = tk.Label(control_frame, text="Your Turn (X)", font=("Arial", 12, "bold"))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)

        # Middle Frame: Tic-Tac-Toe Grid
        grid_frame = tk.Frame(self.root)
        grid_frame.pack()
        
        self.buttons = []
        for i in range(9):
            btn = tk.Button(grid_frame, text="", font=("Arial", 24, "bold"), width=5, height=2,
                            command=lambda idx=i: self.human_move(idx))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

        # Bottom Frame: Reward & Cluster Logs
        log_frame = tk.Frame(self.root, pady=10)
        log_frame.pack()
        
        tk.Label(log_frame, text="Reward & Cluster Log:").pack()
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=10, state='disabled')
        self.log_text.pack()

    def update_log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def train_agent(self):
        self.status_label.config(text="Training... Please wait.")
        self.root.update()
        self.agent.train_self_play(1000)
        self.update_log(self.agent.reward_list[-1])
        self.update_log(f"Clusters created: {len(set(self.agent.state_cluster.values()))}")
        self.status_label.config(text="Training Complete! Your Turn (X)")
        self.reset_game()

    def reset_game(self):
        self.board = np.zeros(9, dtype=int)
        self.agent_history = []
        for btn in self.buttons:
            btn.config(text="", state="normal", bg="SystemButtonFace")
        self.status_label.config(text="Your Turn (X)")

    def disable_board(self):
        for btn in self.buttons:
            btn.config(state="disabled")

    def apply_move(self, action, player, symbol):
        self.board[action] = player
        self.buttons[action].config(text=symbol, state="disabled")
        
        if player == -1: # Agent is O (Red)
            self.buttons[action].config(disabledforeground="red")
        else: # Human is X (Blue)
            self.buttons[action].config(disabledforeground="blue")

    def handle_game_end(self, reward, winner_text):
        self.status_label.config(text=winner_text)
        self.disable_board()
        
        # Agent learns from the human game trajectory
        for state, action in reversed(self.agent_history):
            self.agent.update_q(state, action, reward, tuple(self.board))
            reward = reward * self.agent.gamma # Discount backwards
            
        self.agent.update_clusters()
        
        log_msg = f"Game Over! Reward: {reward:.2f} | Total Clusters: {len(set(self.agent.state_cluster.values()))}"
        self.agent.reward_list.append(log_msg)
        self.update_log(log_msg)

    def human_move(self, action):
        if self.board[action] != 0: return

        # 1. Human Move
        self.apply_move(action, 1, "X")
        reward, done = self.agent.check_winner(self.board)
        
        if done:
            self.handle_game_end(-1 if reward == 1 else 0, "You Win!" if reward == 1 else "It's a Draw!")
            return

        # 2. Agent Move (using Cluster Policy)
        state_tuple = tuple(self.board)
        agent_action = self.agent.choose_cluster_action(state_tuple)
        
        if agent_action is not None:
            self.agent_history.append((state_tuple, agent_action))
            self.apply_move(agent_action, -1, "O")
            
            reward, done = self.agent.check_winner(self.board)
            if done:
                self.handle_game_end(1 if reward == -1 else 0, "Agent Wins!" if reward == -1 else "It's a Draw!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

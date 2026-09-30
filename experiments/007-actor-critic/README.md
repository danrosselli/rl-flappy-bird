# Actor-Critic (Online TD)

## Objective

Replace REINFORCE's episodic training with online Actor-Critic updates to eliminate the memory problem caused by storing entire trajectories. As the bird learns and survives longer, REINFORCE trajectories grow unbounded — Actor-Critic fixes this with O(1) memory per frame.

## Hypothesis

Online one-step TD advantage updates, with a separate actor and critic network, can learn a competent policy without the memory overhead of full-trajectory storage. The critic's baseline reduces gradient variance compared to vanilla REINFORCE.

## Approach

- **State (8-dim)**: horizontal distance and gap alignment for the current and next pipe (`dx`, `dy`, `gap`, `dxNext`, `dyNext`, `gapNext`), bird vertical velocity (`velY`) and current pipe speed (`speed`), all normalized to [-1, +1].
- **Architecture**: Two separate networks sharing the same input:
  - **Actor**: 8→64→64→2 (softmax) — outputs action probabilities
  - **Critic**: 8→64→64→1 (linear) — estimates state value V(s)
- **Update rule**: One-step TD advantage, exactly as the episodic pseudocode in Sutton & Barto §13.1
  - Advantage: A = r + γ·V(s') - V(s) (the TD error δ, unclamped)
  - Actor loss: -log π(a|s) · I · A, where I = γᵗ is the discount-so-far (I ← 1 at episode start, I ← γI after each step; the critic never uses I)
  - Critic loss: 0.5·(target - V(s))² (squared TD error, semi-gradient TD(0))
  - No entropy bonus (A3C), no advantage clipping (DQN-style), no Huber (DQN/Rainbow) — those are not part of the original algorithm.
- **Optimizer**: plain SGD with a fixed step size α (the book's steepest gradient step, not Adam).
- **Training**: Online — one gradient update per frame. No trajectory buffer.
- **Rewards**: pipe passage (+10), collision (-20), velocity penalty (-0.05 when |velY| > 700), flap penalty (-0.05 per flap), and Gaussian proximity shaping (σ=0.5) toward the current gap center. No per-frame survival bonus — same reward structure as 006.
- **Exploration**: Softmax policy sampling (no epsilon-greedy — action probabilities come directly from the actor).

## Key Differences from REINFORCE (006)

| Aspect | REINFORCE (006) | Actor-Critic (007) |
|--------|-----------------|-------------------|
| Memory | O(N) per episode (N = steps) | O(1) per episode |
| Update timing | End of episode | Every frame |
| Variance | High (full returns) | Low (TD advantage) |
| Bias | Unbiased | Biased (critic improves over time) |
| Networks | 1 (policy) | 2 (actor + critic) |
| Optimizer | Adam | Plain SGD (fixed α) |

## Training

Optimizer: plain SGD (steepest gradient step — no momentum, no Adam)
Learning rate (actor): 0.001
Learning rate (critic): 0.002
Gamma: 0.99
Update frequency: every step (online)
Advantage: one-step TD, unclamped (A = δ)
Actor discount: I = γᵗ, reset to 1 on every episode, applied to the actor only
Critic loss: 0.5·MSE on the TD error (no Huber)
Actor loss: -log π(a|s) · I · A (no entropy bonus)

## Fidelity to the original

What the code implements exactly as the book's *One-Step Actor-Critic (Episodic)* pseudocode:

- δ = R + γ·v̂(S') - v̂(S), with v̂(S') = 0 for the terminal state
- w ← w + α^w·δ·∇v̂(S, w)  (critic, semi-gradient, **no** I)
- θ ← θ + α^θ·I·δ·∇ln π(A|S, θ), then I ← γI, with I ← 1 at episode start
- Softmax action sampling for exploration

What is deliberately the project's own choice rather than the book's:

- γ = 0.99 **per frame** (~60 updates/s), so the actor's factor I decays with a half-life of ~69 frames; deep into a long episode the actor step is essentially zero.
- Two hidden layers of 64 ReLU units (the book's figures are tabular/linear).
- Reward shaping (proximity Gaussian, velocity/flap penalties) — the environment, not the algorithm.

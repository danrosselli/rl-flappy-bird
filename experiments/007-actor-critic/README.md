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
- **Update rule**: One-step TD advantage
  - Advantage: A = r + γ·V(s') - V(s), clipped to ±5
  - Actor loss: -log π(a|s) · A - 0.01·H(π)
  - Critic loss: Huber(δ=5) on the TD error
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

## Training

Learning rate (actor): 0.001
Learning rate (critic): 0.002
Gamma: 0.99
Update frequency: every step (online)
Advantage: one-step TD, clipped to ±5
Entropy bonus: 0.01
Critic loss: Huber, δ = 5

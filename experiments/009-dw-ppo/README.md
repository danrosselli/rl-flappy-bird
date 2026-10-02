# Experiment 009 — DW PPO (Death-Window Proximal Policy Optimization)

## Objective

Replace the online Actor-Critic (Experiment 007) with PPO to reduce training noise and improve stability. PPO uses batched updates with a clipped surrogate objective, GAE-lambda advantage estimation, and multiple epochs of mini-batch training per rollout.

## Hypothesis

PPO's clipped objective prevents destructive large policy updates, while GAE-lambda provides lower-variance advantage estimates than one-step TD. Batching 128 steps before updating smooths gradient estimates and produces more stable learning.

## Changes from 007

| Aspect | 007 Actor-Critic | 008 PPO |
|--------|-----------------|---------|
| Update timing | Every frame (online) | Every 128 steps (batched) |
| Advantage | 1-step TD: r + γV(s') - V(s) | GAE-lambda (γ=0.99, λ=0.95) |
| Actor loss | -log π(a|s) · I · A (I = γᵗ) | -min(r·A, clip(r)·A) |
| Critic loss | MSE (0.5·δ²) | MSE(V(s), returns) |
| Optimizer | Plain SGD (fixed α) | Adam |
| Policy update | Unconstrained gradient step | Clipped surrogate (ε=0.2) |
| Gradient clipping | None | Global norm = 0.5 |
| Epochs per update | 1 | 3 |
| Mini-batches | N/A | 4 (size 32) |
| Advantage normalization | None (raw one-step TD) | Z-score normalization |
| Learning rate | 0.001 / 0.002 | 0.0003 / 0.001 |

## Architecture (unchanged)

- Actor: 8 → 64 ReLU → 64 ReLU → 2 softmax
- Critic: 8 → 64 ReLU → 64 ReLU → 1 linear

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Rollout size | 128 |
| PPO epochs | 3 |
| Mini-batch size | 32 |
| Clip epsilon | 0.2 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Max gradient norm | 0.5 |
| Actor learning rate | 0.0003 |
| Critic learning rate | 0.001 |

## Training

Episodes: TBD
Same state representation and reward structure as Experiment 007.

## Results

TBD

## Conclusion

TBD

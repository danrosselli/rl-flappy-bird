/* ============================================================
 * ROLLOUT BUFFER — PPO
 * ------------------------------------------------------------
 * Buffer that collects a fixed number of steps (ROLLOUT_SIZE)
 * before triggering a PPO update. Stores transitions and
 * computes GAE-lambda advantages and discounted returns.
 *
 * Memory layout: contiguous typed arrays for cache efficiency.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';

export const STATE_SIZE = 8;

export class RolloutBuffer {
  /**
   * @param {number} capacity - Maximum number of steps to store
   */
  constructor(capacity = 1024) {
    this.capacity = capacity;
    this.ptr = 0;

    this.states = new Float32Array(capacity * STATE_SIZE);
    this.actions = new Uint8Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.values = new Float32Array(capacity);
    this.logProbs = new Float32Array(capacity);
    this.dones = new Uint8Array(capacity);
    this.overflowWarned = false;
  }

  get size() {
    return this.ptr;
  }

  isReady() {
    return this.ptr >= this.capacity;
  }

  /**
   * Store one transition.
   * @param {number[]} state - 8-dim state vector
   * @param {number} action - 0 (IDLE) or 1 (FLAP)
   * @param {number} reward - Scalar reward
   * @param {number} value  - V(s) from critic
   * @param {number} logProb - log π(a|s) from actor
   * @param {boolean} done  - Whether this is a terminal step
   * @returns {boolean} false if the step was dropped (buffer full)
   */
  add(state, action, reward, value, logProb, done) {
    if (this.ptr >= this.capacity) {
      // Training still running while the buffer refilled (slow machine).
      // Drop the step instead of throwing: the stored rollout is valid and
      // will be consumed by the next startTraining(), then collection
      // resumes normally once the buffer is cleared.
      if (!this.overflowWarned) {
        this.overflowWarned = true;
        console.warn('[RolloutBuffer] full: dropping step (training in progress)');
      }
      return false;
    }

    const offset = this.ptr * STATE_SIZE;
    for (let i = 0; i < STATE_SIZE; i++) {
      this.states[offset + i] = state[i];
    }

    this.actions[this.ptr] = action;
    this.rewards[this.ptr] = reward;
    this.values[this.ptr] = value;
    this.logProbs[this.ptr] = logProb;
    this.dones[this.ptr] = done ? 1 : 0;

    this.ptr++;
    return true;
  }

  /**
   * Compute GAE-lambda advantages and discounted returns.
   *
   * For the final stored transition we need V(s_{T}), i.e. the value
   * of the state AFTER the last stored action. If the last transition
   * is terminal, bootstrapValue must be 0.
   *
   * A_t = δ_t + γλ A_{t+1}
   * δ_t = r_t + γ V(s_{t+1}) - V(s_t)
   *
   * @param {number} bootstrapValue - V(s_T), or 0 when terminal
   * @param {number} gamma - Discount factor
   * @param {number} lam - GAE lambda
   */
  computeAdvantages(bootstrapValue, gamma = 0.99, lam = 0.95) {
    this.advantages = new Float32Array(this.capacity);
    this.returns = new Float32Array(this.capacity);

    let gae = 0;

    for (let t = this.ptr - 1; t >= 0; t--) {
      const nextValue = t === this.ptr - 1 ? bootstrapValue : this.values[t + 1];
      const nextNonTerminal = this.dones[t] ? 0 : 1;

      const delta = this.rewards[t] + gamma * nextValue * nextNonTerminal - this.values[t];
      gae = delta + gamma * lam * nextNonTerminal * gae;

      this.advantages[t] = gae;
      this.returns[t] = gae + this.values[t];
    }
  }

  /**
   * Returns the collected data as tensors for PPO training.
   * Caller must dispose returned tensors.
   *
   * Every tensor is built from a COPY (.slice, never .subarray): tfjs keeps
   * the typed-array reference instead of copying it, so tensors created from
   * buffer memory would alias it — clear() plus the next add() calls would
   * rewrite states and logProbs while training runs across frames.
   *
   * @returns {{ states, actions, advantages, returns, oldLogProbs }}
   */
  get() {
    if (this.ptr === 0) {
      throw new Error('RolloutBuffer is empty');
    }

    if (!this.advantages || !this.returns) {
      throw new Error('RolloutBuffer.get() called before computeAdvantages()');
    }

    const n = this.ptr;

    const states = tf.tensor2d(this.states.slice(0, n * STATE_SIZE), [n, STATE_SIZE]);
    const actions = tf.tensor1d(this.actions.slice(0, n), 'int32');
    const advantages = tf.tensor1d(this.advantages.slice(0, n));
    const returns = tf.tensor1d(this.returns.slice(0, n));
    const oldLogProbs = tf.tensor1d(this.logProbs.slice(0, n));

    return { states, actions, advantages, returns, oldLogProbs };
  }

  /**
   * Clear the buffer for the next rollout.
   */
  clear() {
    this.ptr = 0;
    this.advantages = null;
    this.returns = null;
    this.overflowWarned = false;
  }
}
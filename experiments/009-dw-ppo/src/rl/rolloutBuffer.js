/* ============================================================
 * ROLLING WINDOW BUFFER — DW-PPO (Death-Window PPO)
 * ------------------------------------------------------------
 * Circular buffer that keeps the CAPACITY most recent
 * transitions. When full, each new transition overwrites the
 * oldest one, so the window always ends at the present.
 *
 * Only what is needed to recompute everything at training time
 * is stored: state, action, reward, done. Values (critic) and
 * log-probs (actor) are recomputed with the CURRENT networks
 * right before each update, so they are never stale.
 *
 * Memory layout: contiguous typed arrays.
 * ============================================================ */

export const STATE_SIZE = 8;

export class RolloutBuffer {
  /**
   * @param {number} capacity - Number of most recent steps to keep
   */
  constructor(capacity = 10000) {
    this.capacity = capacity;
    this.head = 0;      // next write position
    this.count = 0;     // valid transitions currently stored

    this.states = new Float32Array(capacity * STATE_SIZE);
    this.actions = new Uint8Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.dones = new Uint8Array(capacity);
  }

  get size() {
    return this.count;
  }

  /**
   * Store one transition, overwriting the oldest one when full.
   * @param {number[]} state - 8-dim state vector
   * @param {number} action - 0 (IDLE) or 1 (FLAP)
   * @param {number} reward - Scalar reward
   * @param {boolean} done  - Whether this is a terminal step
   */
  add(state, action, reward, done) {
    const offset = this.head * STATE_SIZE;
    for (let i = 0; i < STATE_SIZE; i++) {
      this.states[offset + i] = state[i];
    }

    this.actions[this.head] = action;
    this.rewards[this.head] = reward;
    this.dones[this.head] = done ? 1 : 0;

    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  /**
   * Index of the oldest stored transition.
   */
  _start() {
    return this.count < this.capacity ? 0 : this.head;
  }

  /**
   * Copies `arr` (with `stride` values per step) in chronological order,
   * oldest first, into a new typed array of the same type.
   */
  _ordered(arr, stride) {
    const n = this.count;
    const start = this._start();
    const first = Math.min(n, this.capacity - start);
    const second = n - first;

    const out = new arr.constructor(n * stride);
    out.set(arr.subarray(start * stride, (start + first) * stride), 0);

    if (second > 0) {
      out.set(arr.subarray(0, second * stride), first * stride);
    }

    return out;
  }

  /**
   * Returns independent copies of the window in chronological order.
   * Nothing returned aliases the buffer memory.
   * @returns {{ states, actions, rewards, dones, n }}
   */
  getOrdered() {
    if (this.count === 0) {
      throw new Error('RolloutBuffer is empty');
    }

    return {
      states: this._ordered(this.states, STATE_SIZE),
      actions: this._ordered(this.actions, 1),
      rewards: this._ordered(this.rewards, 1),
      dones: this._ordered(this.dones, 1),
      n: this.count
    };
  }

  clear() {
    this.head = 0;
    this.count = 0;
  }
}

/**
 * GAE-lambda advantages and returns over a chronologically ordered window.
 *
 *   δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)
 *   A_t = δ_t + γλ (1 - done_t) A_{t+1}
 *
 * `done` cuts the recursion, so episodes that share the window never leak
 * into each other. The oldest episode may be cut at the start of the window:
 * that is harmless, since GAE only looks forward in time.
 *
 * @param {Float32Array} rewards
 * @param {Float32Array} values - V(s_t) from the current critic
 * @param {Uint8Array} dones
 * @param {number} lastValue - V(s_T) after the last step (0 when terminal)
 * @returns {{ advantages: Float32Array, returns: Float32Array }}
 */
export function computeGAE(rewards, values, dones, lastValue, gamma, lam) {
  const n = rewards.length;
  const advantages = new Float32Array(n);
  const returns = new Float32Array(n);

  let gae = 0;

  for (let t = n - 1; t >= 0; t--) {
    const nextValue = t === n - 1 ? lastValue : values[t + 1];
    const nonTerminal = dones[t] ? 0 : 1;

    const delta = rewards[t] + gamma * nextValue * nonTerminal - values[t];
    gae = delta + gamma * lam * nonTerminal * gae;

    advantages[t] = gae;
    returns[t] = gae + values[t];
  }

  return { advantages, returns };
}
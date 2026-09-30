/* ============================================================
 * ACTOR-CRITIC (Online TD) — FLAPPY BIRD
 * ------------------------------------------------------------
 * The Phaser game is the training environment itself.
 *
 * Two separate networks:
 *   - Actor:  outputs action probabilities (softmax)
 *   - Critic: estimates state value V(s) (linear output)
 *
 * Updates happen online — one gradient step per frame.
 * No trajectory buffer. Memory usage is O(1) regardless of
 * episode length, solving the memory issue of vanilla REINFORCE.
 *
 * Follows the one-step Actor-Critic of Sutton & Barto §13.5, with
 * two deliberate deviations: the discount-so-far factor I = γ^t of
 * the episodic pseudocode is omitted from the actor update, as in
 * most practical implementations, and an entropy bonus and advantage
 * clipping are added to the actor update. The one-step TD error
 * δ = r + γ * V(s') - V(s) is the single critique signal that
 * trains both networks.
 *
 * Advantage: A = δ = r + γ * V(s') - V(s)  (one-step TD error, clipped)
 * Actor loss:  -log π(a|s) * A - ENTROPY_COEF * H(π)
 * Critic loss: 0.5 * (target - V(s))^2  (semi-gradient TD(0))
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { PersistenceManager } from './persistenceManager.js';

export const ACTION_IDLE = 0;
export const ACTION_FLAP = 1;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];
export const STATE_SIZE = 8;
export const ACTION_SIZE = 2;

export const gamma = 0.99;
export const LEARNING_RATE = 0.001;
export const CRITIC_LEARNING_RATE = 0.002;

const POLICY_EPSILON = 1e-7;

// Entropy bonus: keeps the policy from collapsing to a deterministic
// action too early. Without this, softmax probabilities can saturate
// near 0/1 and gradients vanish, making the collapse irreversible.
const ENTROPY_COEF = 0.01;

// Clamps the TD advantage before it drives the actor's gradient step.
// A single large advantage (e.g. from the -20 death reward) can
// otherwise push probabilities to the extremes in one update.
// The critic still trains on the true, unclipped target.
const ADVANTAGE_CLIP = 5;

// Single source of truth for the critic objective: it is both the loss
// passed to critic.compile() and the function minimized in update().
// 0.5 * MSE is the squared TD error of the classic semi-gradient
// update w <- w + alpha * delta * grad v(s; w).
function mseLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const error = yTrue.sub(yPred);
    return error.square().mul(0.5).mean();
  });
}

export class ActorCriticAgent {
  constructor() {
    this.actor = this.createActor();
    this.critic = this.createCritic();
    this.persistence = new PersistenceManager();
    this.trainingInProgress = false;
    this.lastTrainingLoss = null;
    this.lastCriticLoss = null;
  }

  createActor() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [STATE_SIZE]
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    model.add(tf.layers.dense({
      units: ACTION_SIZE,
      activation: 'softmax'
    }));

    // compile() is what attaches an optimizer to the model; the objective
    // actually optimized here is the policy gradient applied by update(),
    // not the cross-entropy declared below (which model.fit() would use).
    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'categoricalCrossentropy'
    });

    return model;
  }

  createCritic() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [STATE_SIZE]
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    model.add(tf.layers.dense({
      units: 1,
      activation: 'linear'
    }));

    model.compile({
      optimizer: tf.train.adam(CRITIC_LEARNING_RATE),
      loss: mseLoss
    });

    return model;
  }

  /**
   * Samples an action from the actor and estimates V(s) from the critic.
   * Returns action, action probabilities, and value estimate.
   */
  chooseAction(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);

      // Actor forward pass
      const probabilities = this.actor.predict(stateTensor);
      const probs = probabilities.dataSync();

      // Sample action
      const random = Math.random();
      const action = random < probs[ACTION_FLAP]
        ? ACTION_FLAP
        : ACTION_IDLE;

      // Critic forward pass
      const valueTensor = this.critic.predict(stateTensor);
      const value = valueTensor.dataSync()[0];

      return {
        action,
        idleProbability: probs[ACTION_IDLE],
        flapProbability: probs[ACTION_FLAP],
        value
      };
    });
  }

  /**
   * Online Actor-Critic update (one-step TD) — Sutton & Barto §13.5,
   * without the discount-so-far factor I.
   *
   * Called every frame after the action is executed and the reward
   * is observed, along with the next state.
   *
   * 1. Compute advantage: A = r + γ * V(s') - V(s)   (the TD error δ, clipped for the actor)
   * 2. Update critic:  minimize 0.5 * (target - V(s))^2   (semi-gradient)
   * 3. Update actor:   minimize -log π(a|s) * A - ENTROPY_COEF * H(π)
   */
  async update(prevState, prevAction, prevValue, reward, nextState, done) {
    if (this.trainingInProgress) {
      return null;
    }

    this.trainingInProgress = true;

    // Declared outside the try so the finally can free them even when
    // something throws between creation and the normal dispose().
    let stateTensor = null;
    let nextStateTensor = null;
    let valueCurrentTensor = null;
    let criticLoss = null;
    let actorLoss = null;

    try {
      stateTensor = tf.tensor2d([prevState], [1, STATE_SIZE]);
      nextStateTensor = tf.tensor2d([nextState], [1, STATE_SIZE]);

      // V(s') for advantage calculation. predict() returns a tensor the
      // caller owns — it must be disposed explicitly, otherwise it leaks
      // one tensor per frame. (dispose() is idempotent, the finally
      // below is the safety net if dataSync() throws.)
      valueCurrentTensor = this.critic.predict(nextStateTensor);
      const valueCurrent = valueCurrentTensor.dataSync()[0];
      valueCurrentTensor.dispose();
      const target = reward + (done ? 0 : gamma * valueCurrent);
      let advantage = target - prevValue;
      advantage = Math.max(-ADVANTAGE_CLIP, Math.min(ADVANTAGE_CLIP, advantage));

      // --- Critic update: minimize 0.5 * (target - V(s))^2 ---
      // minimize() scopes gradients to only this model's variables.
      // `target` is computed outside the graph, so this is a
      // semi-gradient step — exactly w <- w + a * delta * grad v(s; w).
      criticLoss = this.critic.optimizer.minimize(() => {
        const v = this.critic.predict(stateTensor);
        const targetTensor = tf.tensor2d([[target]], [1, 1]);
        return mseLoss(targetTensor, v);
      }, true);

      // --- Actor update: minimize -log π(a|s) * advantage - entropyCoef * H(π) ---
      // The entropy term H(π) = -Σ p·log(p) rewards keeping some spread
      // across actions, so the policy doesn't lock onto one action forever.
      actorLoss = this.actor.optimizer.minimize(() => {
        const probs = this.actor.predict(stateTensor);
        const safeProbs = probs.add(POLICY_EPSILON);
        const logProbs = safeProbs.log();
        const actionMask = tf.oneHot(tf.tensor1d([prevAction], 'int32'), ACTION_SIZE);
        const selectedLogProb = logProbs.mul(actionMask).sum(1);

        const entropy = safeProbs.mul(logProbs).sum(1).mul(-1);

        return selectedLogProb.mul(-advantage)
          .sub(entropy.mul(ENTROPY_COEF))
          .mean();
      }, true);

      this.lastTrainingLoss = actorLoss.dataSync()[0];
      this.lastCriticLoss = criticLoss.dataSync()[0];

      return this.lastTrainingLoss;
    } catch (error) {
      console.error('Actor-Critic update error:', error);
      return null;
    } finally {
      // Free everything unconditionally: an exception in the middle of
      // the update must not leak the state tensors or the loss tensors
      // (they are created outside minimize()'s internal tidy scope).
      if (valueCurrentTensor) valueCurrentTensor.dispose();
      if (actorLoss) actorLoss.dispose();
      if (criticLoss) criticLoss.dispose();
      if (stateTensor) stateTensor.dispose();
      if (nextStateTensor) nextStateTensor.dispose();
      this.trainingInProgress = false;
    }
  }

  getPolicy(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const probabilities = this.actor.predict(stateTensor);
      const p = probabilities.dataSync();

      return {
        [ACTION_IDLE]: p[ACTION_IDLE],
        [ACTION_FLAP]: p[ACTION_FLAP]
      };
    });
  }

  getValue(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const v = this.critic.predict(stateTensor);
      return v.dataSync()[0];
    });
  }

  async saveBrain(generation, highScore = 0) {
    await this.persistence.saveActor(this.actor);
    await this.persistence.saveCritic(this.critic);
    await this.persistence.saveMetadata({
      algorithm: 'actor-critic',
      generation,
      highScore
    });
    console.log('Actor-Critic salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const metadata = await this.persistence.loadMetadata();

      if (metadata && metadata.algorithm && metadata.algorithm !== 'actor-critic') {
        console.log('Memória antiga pertence a outro algoritmo. Iniciando Actor-Critic.');
        await this.persistence.clearAll();
        return { success: false, generation: 1 };
      }

      const actor = await this.persistence.loadActor();
      const critic = await this.persistence.loadCritic();

      // A partial save (only one of the two networks written) can happen
      // if the page closes between saveActor() and saveCritic(). Such a
      // pair can never be resumed — drop the orphan instead of keeping it.
      if ((actor == null) !== (critic == null)) {
        console.log('Save parcial (actor/critic incompletos). Resetando.');
        await this.persistence.clearAll();
        return { success: false, generation: 1 };
      }

      if (actor && critic) {
        const actorInputSize = actor.inputs[0].shape[1];
        const actorOutputSize = actor.outputs[0].shape[1];
        const criticInputSize = critic.inputs[0].shape[1];
        const criticOutputSize = critic.outputs[0].shape[1];

        if (actorInputSize !== STATE_SIZE || actorOutputSize !== ACTION_SIZE ||
            criticInputSize !== STATE_SIZE || criticOutputSize !== 1) {
          console.log('Modelos salvos incompatíveis com Actor-Critic. Resetando.');
          await this.persistence.clearAll();
          return { success: false, generation: 1 };
        }

        this.actor = actor;
        this.actor.compile({
          optimizer: tf.train.adam(LEARNING_RATE),
          loss: 'categoricalCrossentropy'
        });

        this.critic = critic;
        this.critic.compile({
          optimizer: tf.train.adam(CRITIC_LEARNING_RATE),
          loss: mseLoss
        });

        if (metadata) {
          const generation = metadata.generation ?? 1;
          const highScore = metadata.highScore ?? 0;
          console.log(
            'Actor-Critic carregado. Gen:',
            generation,
            'HighScore:',
            highScore
          );
          return { success: true, generation, highScore };
        }

        return { success: true, generation: 1, highScore: 0 };
      }
    } catch (error) {
      console.log('Nenhum estado de Actor-Critic salvo encontrado:', error.message);
    }

    return { success: false, generation: 1 };
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  console.log('Memória do Actor-Critic resetada');
}
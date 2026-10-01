/* ============================================================
 * PPO (Proximal Policy Optimization) — FLAPPY BIRD
 * ------------------------------------------------------------
 * Two separate networks:
 *   - Actor:  outputs action probabilities (softmax)
 *   - Critic: estimates state value V(s) (linear output)
 *
 * Training is batched: collect ROLLOUT_SIZE steps, then run
 * PPO_EPOCHS passes over the data with clipped surrogate
 * objective and GAE-lambda advantages.
 *
 * Advantage: GAE-lambda (low-variance advantage estimation)
 * Actor loss: -min(r·A, clip(r,1-ε,1+ε)·A) - c_ent·H(π)
 * Critic loss: MSE(V(s), returns)
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { RolloutBuffer, STATE_SIZE } from './rolloutBuffer.js';
import { PersistenceManager } from './persistenceManager.js';

export const ACTION_IDLE = 0;
export const ACTION_FLAP = 1;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];
export { STATE_SIZE };
export const ACTION_SIZE = 2;

// --- PPO Hyperparameters ---
export const ROLLOUT_SIZE = 1024;
const PPO_EPOCHS = 3;
const MINI_BATCH_SIZE = 128;
const CLIP_EPSILON = 0.2;
const GAMMA = 0.99;
const LAMBDA = 0.95;
const ENTROPY_COEF = 0.01;
const VALUE_LOSS_COEF = 0.5;
const ACTOR_LR = 3e-4;
const CRITIC_LR = 1e-3;
const POLICY_EPSILON = 1e-7;
const MIN_BATCH_SIZE = 32;
const MAX_GRAD_NORM = 0.5;

const nextFrame = () => new Promise(res => requestAnimationFrame(res));

function clipGradsByGlobalNorm(grads, maxNorm) {
  const tensors = Object.values(grads);

  const maxNormScalar = tf.scalar(maxNorm);
  const gradSquares = tensors.map(t => t.square().sum());
  const globalNorm = tf.addN(gradSquares).sqrt();
  const clipCoeff = tf.minimum(globalNorm, maxNormScalar).div(globalNorm.add(1e-6));

  const clipped = {};
  for (const [key, grad] of Object.entries(grads)) {
    clipped[key] = grad.mul(clipCoeff);
  }

  // Dispose intermediate tensors (NOT the input grads or the clipped results).
  gradSquares.forEach(t => t.dispose());
  globalNorm.dispose();
  maxNormScalar.dispose();
  clipCoeff.dispose();

  return clipped;
}

export class PPOAgent {
  constructor() {
    this.actor = this.createActor();
    this.critic = this.createCritic();
    this.actorOptimizer = tf.train.adam(ACTOR_LR);
    this.criticOptimizer = tf.train.adam(CRITIC_LR);
    this.buffer = new RolloutBuffer(ROLLOUT_SIZE);
    this.persistence = new PersistenceManager();

    this.lastActorLoss = null;
    this.lastCriticLoss = null;
    this.lastClipFraction = null;
    this.trainingInProgress = false;
    this.trainingPromise = null;
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

    return model;
  }

  /**
   * Samples an action from the actor and estimates V(s) from the critic.
   */
  chooseAction(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);

      const probabilities = this.actor.predict(stateTensor);
      const probs = probabilities.dataSync();

      const random = Math.random();
      const action = random < probs[ACTION_FLAP] ? ACTION_FLAP : ACTION_IDLE;

      const logProb = Math.log(probs[action] + POLICY_EPSILON);
      const valueTensor = this.critic.predict(stateTensor);
      const value = valueTensor.dataSync()[0];

      return {
        action,
        idleProbability: probs[ACTION_IDLE],
        flapProbability: probs[ACTION_FLAP],
        logProb,
        value
      };
    });
  }

  /**
   * Start PPO training and keep a reference to the active Promise.
   */
  startTraining(lastValue) {
    if (this.trainingInProgress) return this.trainingPromise;

    const promise = this.train(lastValue);
    this.trainingPromise = promise;

    // Note: do NOT chain .finally() here, which would create an untracked
    // child promise. Instead, the caller awaits the returned promise and the
    // cleanup is done inside train()'s finally block.
    return promise;
  }

  /**
   * Collect one transition.
   *
   * bootstrapValue is the value of the NEXT state after this transition.
   * It is only needed when this step closes a non-terminal rollout.
   */
  collectStep(state, action, reward, logProb, value, done, bootstrapValue = 0) {
    this.buffer.add(state, action, reward, value, logProb, done);

    if (this.buffer.isReady()) {
      return this.startTraining(done ? 0 : bootstrapValue);
    }

    return null;
  }

  /**
   * Force a PPO update with the current buffer content.
   * At a terminal episode boundary, pass 0.
   * For a non-terminal partial rollout, pass V(nextState).
   */
  async forceUpdate(bootstrapValue = 0) {
    if (this.trainingInProgress && this.trainingPromise) {
      await this.trainingPromise;
    }

    if (this.buffer.size === 0) return null;
    if (this.buffer.size < MIN_BATCH_SIZE) return null;

    return this.startTraining(bootstrapValue);
  }

  /**
   * Core PPO training loop.
   * 1. Compute GAE-lambda advantages and discounted returns
   * 2. Run PPO_EPOCHS passes, each split into mini-batches
   * 3. Clipped surrogate objective + value loss
   */
  async train(lastValue) {
    if (this.trainingInProgress) return null;
    this.trainingInProgress = true;

    let states = null, actions = null, advantages = null,
        returns = null, oldLogProbs = null, normalizedAdvantages = null;
    let consumed = false;

    try {
      // 1. Compute advantages and returns
      this.buffer.computeAdvantages(lastValue, GAMMA, LAMBDA);

      // 2. Get data as tensors
      ({ states, actions, advantages, returns, oldLogProbs } = this.buffer.get());
      const batchSize = this.buffer.size;

      // get() builds tensors from copies of the buffer memory (see
      // RolloutBuffer.get), so the buffer can be freed immediately while the
      // training runs in time slices across frames.
      this.buffer.clear();
      consumed = true;

      // Normalize advantages per rollout (zero mean, unit std of THIS batch)
      const rawAdvantages = Array.from(advantages.dataSync());
      const advMean = rawAdvantages.reduce((s, v) => s + v, 0) / batchSize;
      const advStd = Math.sqrt(
        rawAdvantages.reduce((s, v) => s + (v - advMean) ** 2, 0) / batchSize
      );
      normalizedAdvantages = tf.tensor1d(
        rawAdvantages.map(v => (v - advMean) / (advStd + 1e-8))
      );

      // Shuffle independently for every PPO epoch.
      const indices = Array.from({ length: batchSize }, (_, i) => i);

      let totalActorLoss = 0;
      let totalCriticLoss = 0;
      let totalClipFrac = 0;
      let numBatches = 0;

      // 3. PPO epochs
      for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        for (let start = 0; start < batchSize; start += MINI_BATCH_SIZE) {
          const end = Math.min(start + MINI_BATCH_SIZE, batchSize);

          await nextFrame();

          // Entire mini-batch enclosed in tf.tidy so ALL intermediate tensors
          // (gather results, forward-pass activations, gradients and clipped
          // gradients) are disposed automatically. Only JS numbers escape.
          const { aLoss, cLoss, clipFrac } = tf.tidy(() => {
            const sliceIdx = indices.slice(start, end);
            const idxTensor = tf.tensor1d(sliceIdx, 'int32');

            const miniStates = tf.gather(states, idxTensor);
            const miniActions = tf.gather(actions, idxTensor);
            const miniAdvantages = tf.gather(normalizedAdvantages, idxTensor);
            const miniOldLogProbs = tf.gather(oldLogProbs, idxTensor);
            const miniReturns = tf.gather(returns, idxTensor);
            idxTensor.dispose();

            // --- Actor update (with gradient clipping) ---
            const actorResult = this.actorOptimizer.computeGradients(() => {
              const probs = this.actor.predict(miniStates);
              const safeProbs = probs.add(POLICY_EPSILON);
              const logProbsAll = safeProbs.log();
              const actionMask = tf.oneHot(miniActions, ACTION_SIZE);
              const newLogProbs = logProbsAll.mul(actionMask).sum(1);

              const ratio = newLogProbs.sub(miniOldLogProbs).exp();
              const clippedRatio = tf.clipByValue(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON);
              const surr1 = ratio.mul(miniAdvantages);
              const surr2 = clippedRatio.mul(miniAdvantages);
              const policyLoss = tf.minimum(surr1, surr2).mean().neg();

              const entropy = safeProbs.mul(logProbsAll).sum(1).neg().mean();

              return policyLoss.sub(entropy.mul(ENTROPY_COEF));
            });
            this.actorOptimizer.applyGradients(
              clipGradsByGlobalNorm(actorResult.grads, MAX_GRAD_NORM)
            );
            const aLoss = actorResult.value.dataSync()[0];

            // Compute clip fraction for monitoring (shared forward pass)
            const aProbs = this.actor.predict(miniStates);
            const aSafeProbs = aProbs.add(POLICY_EPSILON);
            const aLogProbsAll = aSafeProbs.log();
            const aMask = tf.oneHot(miniActions, ACTION_SIZE);
            const aNewLogProbs = aLogProbsAll.mul(aMask).sum(1);
            const aRatio = aNewLogProbs.sub(miniOldLogProbs).exp();
            const aClipped = aRatio.less(1 - CLIP_EPSILON).logicalOr(aRatio.greater(1 + CLIP_EPSILON));
            const clipFrac = aClipped.toFloat().mean().dataSync()[0];

            // --- Critic update (with gradient clipping) ---
            const criticResult = this.criticOptimizer.computeGradients(() => {
              const values = this.critic.predict(miniStates).squeeze();
              const valueLoss = values.sub(miniReturns).square().mean();
              return valueLoss.mul(VALUE_LOSS_COEF);
            });
            this.criticOptimizer.applyGradients(
              clipGradsByGlobalNorm(criticResult.grads, MAX_GRAD_NORM)
            );
            const cLoss = criticResult.value.dataSync()[0];

            return { aLoss, cLoss, clipFrac };
          });

          totalActorLoss += aLoss;
          totalCriticLoss += cLoss;
          totalClipFrac += clipFrac;
          numBatches++;
        }
      }

      this.lastActorLoss = totalActorLoss / numBatches;
      this.lastCriticLoss = totalCriticLoss / numBatches;
      this.lastClipFraction = totalClipFrac / numBatches;
      if (tf.engine().backend) {
        const mem = tf.memory();
        console.log(`[PPO] train done | tensors: ${mem.numTensors} | MB: ${(mem.numBytes / 1048576).toFixed(2)}`);
      }

      return this.lastActorLoss;
    } catch (error) {
      console.error('PPO training error:', error);
      return null;
    } finally {
      // Guarantee cleanup of all rollout tensors even on error.
      // If computeAdvantages() or get() threw, the rollout is still sitting
      // in the buffer: drop it so a full buffer can never block add().
      if (!consumed) this.buffer.clear();

      try { if (states) states.dispose(); } catch (e) { }
      try { if (actions) actions.dispose(); } catch (e) { }
      try { if (advantages) advantages.dispose(); } catch (e) { }
      try { if (normalizedAdvantages) normalizedAdvantages.dispose(); } catch (e) { }
      try { if (returns) returns.dispose(); } catch (e) { }
      try { if (oldLogProbs) oldLogProbs.dispose(); } catch (e) { }

      this.trainingInProgress = false;
      this.trainingPromise = null;
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
    // Wait for any in-flight PPO update: model.save() reads the weights
    // asynchronously, and applyGradients() replaces them in the meantime,
    // which could save inconsistent weights or make the save fail.
    if (this.trainingPromise) {
      await this.trainingPromise;
    }

    await this.persistence.saveActor(this.actor);
    await this.persistence.saveCritic(this.critic);
    await this.persistence.saveMetadata({
      algorithm: 'ppo',
      generation,
      highScore
    });

    console.log('PPO salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const metadata = await this.persistence.loadMetadata();

      if (metadata && metadata.algorithm && metadata.algorithm !== 'ppo') {
        console.log('Memória antiga pertence a outro algoritmo. Iniciando PPO.');
        await this.persistence.clearAll();
        return { success: false, generation: 1 };
      }

      const actor = await this.persistence.loadActor();
      const critic = await this.persistence.loadCritic();

      if (actor && critic) {
        const actorInputSize = actor.inputs[0].shape[1];
        const actorOutputSize = actor.outputs[0].shape[1];
        const criticInputSize = critic.inputs[0].shape[1];
        const criticOutputSize = critic.outputs[0].shape[1];

        if (actorInputSize !== STATE_SIZE || actorOutputSize !== ACTION_SIZE ||
            criticInputSize !== STATE_SIZE || criticOutputSize !== 1) {
          console.log('Modelos salvos incompatíveis com PPO. Resetando.');
          await this.persistence.clearAll();
          return { success: false, generation: 1 };
        }

        this.actor = actor;
        this.critic = critic;
        this.actorOptimizer = tf.train.adam(ACTOR_LR);
        this.criticOptimizer = tf.train.adam(CRITIC_LR);

        if (metadata) {
          const generation = metadata.generation ?? 1;
          const highScore = metadata.highScore ?? 0;

          console.log(
            'PPO carregado. Gen:',
            generation,
            'HighScore:',
            highScore
          );

          return { success: true, generation, highScore };
        }

        return { success: true, generation: 1, highScore: 0 };
      }
    } catch (error) {
      console.log('Nenhum estado PPO salvo encontrado:', error.message);
    }

    return { success: false, generation: 1 };
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  console.log('Memória do PPO resetada');
}
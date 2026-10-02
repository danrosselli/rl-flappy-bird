/* ============================================================
 * DW-PPO (Death-Window PPO) — FLAPPY BIRD
 * ------------------------------------------------------------
 * PPO variant for continual learning:
 *   - Transitions go into a rolling window with the BUFFER_CAPACITY
 *     most recent steps (or fewer, if the agent dies early). After each
 *     update the window is discarded.
 *   - Training happens ONLY when the bird dies. While it survives,
 *     the policy stays frozen, so the window is fully on-policy.
 *   - Right before each update, values V(s) and old log-probs are
 *     recomputed with the CURRENT critic/actor and GAE is rebuilt
 *     over the window. The ratio therefore starts at 1 and the
 *     clipping only limits movement inside this update.
 *   - The last step of the window is always terminal (the death),
 *     so the bootstrap value is exactly 0.
 *
 * Two separate networks:
 *   - Actor:  outputs action probabilities (softmax)
 *   - Critic: estimates state value V(s) (linear output)
 *
 * Actor loss: -min(r·A, clip(r,1-ε,1+ε)·A) - c_ent·H(π)
 * Critic loss: MSE(V(s), returns)
 * Early stop: the update ends when approx KL exceeds 1.5·TARGET_KL.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { RolloutBuffer, computeGAE, STATE_SIZE } from './rolloutBuffer.js';
import { PersistenceManager } from './persistenceManager.js';

export const ACTION_IDLE = 0;
export const ACTION_FLAP = 1;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];
export { STATE_SIZE };
export const ACTION_SIZE = 2;

const ALGORITHM = 'dw-ppo';

// --- DW-PPO Hyperparameters ---
export const BUFFER_CAPACITY = 10000;   // most recent steps kept in the window
const PPO_EPOCHS = 3;
const MINI_BATCH_SIZE = 512;
const TARGET_KL = 0.02;                 // stop when approx KL > 1.5 * TARGET_KL
const CLIP_EPSILON = 0.2;
const GAMMA = 0.99;
const LAMBDA = 0.97;
const ENTROPY_COEF = 0.01;
const VALUE_LOSS_COEF = 0.5;
const ACTOR_LR = 3e-4;
const CRITIC_LR = 1e-3;
const POLICY_EPSILON = 1e-7;
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
    this.buffer = new RolloutBuffer(BUFFER_CAPACITY);
    this.persistence = new PersistenceManager();

    this.lastActorLoss = null;
    this.lastCriticLoss = null;
    this.lastClipFraction = null;
    this.lastKL = null;
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
   * Samples an action from the actor. V(s) is returned only for the HUD:
   * training recomputes all values from the stored states.
   */
  chooseAction(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);

      const probs = this.actor.predict(stateTensor).dataSync();

      const action = Math.random() < probs[ACTION_FLAP] ? ACTION_FLAP : ACTION_IDLE;

      const value = this.critic.predict(stateTensor).dataSync()[0];

      return {
        action,
        idleProbability: probs[ACTION_IDLE],
        flapProbability: probs[ACTION_FLAP],
        value
      };
    });
  }

  /**
   * Stores one transition in the rolling window. Never trains by itself:
   * training is triggered by trainOnDeath().
   */
  collectStep(state, action, reward, done) {
    this.buffer.add(state, action, reward, done);
  }

  /**
   * Run a PPO update over the current window. Call it right after storing
   * the terminal (death) transition. Returns the training promise.
   */
  trainOnDeath() {
    if (this.trainingInProgress) return this.trainingPromise;
    if (this.buffer.size === 0) return Promise.resolve(null);

    const promise = this.train();
    this.trainingPromise = promise;
    return promise;
  }

  /**
   * Core DW-PPO training:
   * 1. Recompute V(s) and old log-probs with the current networks
   * 2. GAE-lambda over the window (last step is terminal => bootstrap 0)
   * 3. PPO epochs over mini-batches, with KL early stopping
   */
  async train() {
    if (this.trainingInProgress) return null;
    this.trainingInProgress = true;

    let states = null, actions = null, advantages = null,
        returns = null, oldLogProbs = null;

    try {
      const data = this.buffer.getOrdered();
      const n = data.n;

      // The window is always discarded after being consumed (copies above).
      this.buffer.clear();

      // 1. Values and old log-probs from the CURRENT networks (one pass).
      const actionsInt = Int32Array.from(data.actions);
      let values = null, oldLogProbsArr = null;

      tf.tidy(() => {
        const s = tf.tensor2d(data.states, [n, STATE_SIZE]);
        const a = tf.tensor1d(actionsInt, 'int32');

        values = this.critic.predict(s).reshape([-1]).dataSync();

        const probs = this.actor.predict(s);
        const logProbsAll = probs.add(POLICY_EPSILON).log();
        oldLogProbsArr = logProbsAll.mul(tf.oneHot(a, ACTION_SIZE)).sum(1).dataSync();
      });

      // 2. GAE over the window. The last step is the death (done = true).
      const gae = computeGAE(data.rewards, values, data.dones, 0, GAMMA, LAMBDA);

      // Normalize advantages over this window (zero mean, unit std).
      let advMean = 0;
      for (let i = 0; i < n; i++) advMean += gae.advantages[i];
      advMean /= n;

      let advVar = 0;
      for (let i = 0; i < n; i++) advVar += (gae.advantages[i] - advMean) ** 2;
      const advStd = Math.sqrt(advVar / n);

      const normAdv = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        normAdv[i] = (gae.advantages[i] - advMean) / (advStd + 1e-8);
      }

      states = tf.tensor2d(data.states, [n, STATE_SIZE]);
      actions = tf.tensor1d(actionsInt, 'int32');
      advantages = tf.tensor1d(normAdv);
      returns = tf.tensor1d(gae.returns);
      oldLogProbs = tf.tensor1d(oldLogProbsArr);

      // Even-sized mini-batches (no tiny leftover batch).
      const numMiniBatches = Math.ceil(n / MINI_BATCH_SIZE);
      const batchSize = Math.ceil(n / numMiniBatches);
      const indices = Array.from({ length: n }, (_, i) => i);

      let totalActorLoss = 0;
      let totalCriticLoss = 0;
      let totalClipFrac = 0;
      let lastKL = 0;
      let numBatches = 0;
      let stop = false;

      // 3. PPO epochs
      for (let epoch = 0; epoch < PPO_EPOCHS && !stop; epoch++) {
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        for (let start = 0; start < n; start += batchSize) {
          const end = Math.min(start + batchSize, n);

          await nextFrame();

          // Entire mini-batch enclosed in tf.tidy so ALL intermediate tensors
          // are disposed automatically. Only JS numbers escape.
          const { aLoss, cLoss, clipFrac, kl } = tf.tidy(() => {
            const idxTensor = tf.tensor1d(indices.slice(start, end), 'int32');

            const miniStates = tf.gather(states, idxTensor);
            const miniActions = tf.gather(actions, idxTensor);
            const miniAdvantages = tf.gather(advantages, idxTensor);
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

            // Monitoring after the step: clip fraction and approx KL (k3).
            const aProbs = this.actor.predict(miniStates);
            const aSafeProbs = aProbs.add(POLICY_EPSILON);
            const aLogProbsAll = aSafeProbs.log();
            const aMask = tf.oneHot(miniActions, ACTION_SIZE);
            const aNewLogProbs = aLogProbsAll.mul(aMask).sum(1);
            const aLogRatio = aNewLogProbs.sub(miniOldLogProbs);
            const aRatio = aLogRatio.exp();
            const aClipped = aRatio.less(1 - CLIP_EPSILON).logicalOr(aRatio.greater(1 + CLIP_EPSILON));
            const clipFrac = aClipped.toFloat().mean().dataSync()[0];
            const kl = aRatio.sub(1).sub(aLogRatio).mean().dataSync()[0];

            // --- Critic update (with gradient clipping) ---
            const criticResult = this.criticOptimizer.computeGradients(() => {
              const values = this.critic.predict(miniStates).reshape([-1]);
              const valueLoss = values.sub(miniReturns).square().mean();
              return valueLoss.mul(VALUE_LOSS_COEF);
            });
            this.criticOptimizer.applyGradients(
              clipGradsByGlobalNorm(criticResult.grads, MAX_GRAD_NORM)
            );
            const cLoss = criticResult.value.dataSync()[0];

            return { aLoss, cLoss, clipFrac, kl };
          });

          totalActorLoss += aLoss;
          totalCriticLoss += cLoss;
          totalClipFrac += clipFrac;
          lastKL = kl;
          numBatches++;

          // KL early stop: the policy already moved enough in this update.
          if (kl > 1.5 * TARGET_KL) {
            stop = true;
            break;
          }
        }
      }

      this.lastActorLoss = totalActorLoss / numBatches;
      this.lastCriticLoss = totalCriticLoss / numBatches;
      this.lastClipFraction = totalClipFrac / numBatches;
      this.lastKL = lastKL;

      const mem = tf.memory();
      console.log(
        `[DW-PPO] train done | steps: ${n} | minibatches: ${numBatches}` +
        `${stop ? ' (KL stop)' : ''} | KL: ${lastKL.toFixed(4)}` +
        ` | tensors: ${mem.numTensors} | MB: ${(mem.numBytes / 1048576).toFixed(2)}`
      );

      return this.lastActorLoss;
    } catch (error) {
      console.error('DW-PPO training error:', error);
      return null;
    } finally {
      try { if (states) states.dispose(); } catch (e) { }
      try { if (actions) actions.dispose(); } catch (e) { }
      try { if (advantages) advantages.dispose(); } catch (e) { }
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
    // Wait for any in-flight update: model.save() reads the weights
    // asynchronously, and applyGradients() replaces them in the meantime.
    if (this.trainingPromise) {
      await this.trainingPromise;
    }

    await this.persistence.saveActor(this.actor);
    await this.persistence.saveCritic(this.critic);
    await this.persistence.saveMetadata({
      algorithm: ALGORITHM,
      generation,
      highScore
    });

    console.log('DW-PPO salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const metadata = await this.persistence.loadMetadata();

      if (metadata && metadata.algorithm && metadata.algorithm !== ALGORITHM) {
        console.log('Memória antiga pertence a outro algoritmo. Iniciando DW-PPO.');
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
          console.log('Modelos salvos incompatíveis com DW-PPO. Resetando.');
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
            'DW-PPO carregado. Gen:',
            generation,
            'HighScore:',
            highScore
          );

          return { success: true, generation, highScore };
        }

        return { success: true, generation: 1, highScore: 0 };
      }
    } catch (error) {
      console.log('Nenhum estado DW-PPO salvo encontrado:', error.message);
    }

    return { success: false, generation: 1 };
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  console.log('Memória do DW-PPO resetada');
}
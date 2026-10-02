/* ============================================================
 * PERSISTENCE MANAGER — PPO
 * ------------------------------------------------------------
 * IndexedDB persistence for actor and critic networks.
 * Uses experiment-specific keys to avoid conflicts with
 * other experiments (e.g. 007-actor-critic).
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import experimentConfig from '../../experiment.json';

const EXPERIMENT_NAME =
  (experimentConfig && experimentConfig.name)
    ? experimentConfig.name
    : 'experiment';

const DB_NAME = `FlappyPPODB_${EXPERIMENT_NAME}`;
const DB_VERSION = 1;
const GAME_DATA_STORE = 'gameData';

export class PersistenceManager {
  constructor() {
    this.actorPath =
      `indexeddb://flappy-ppo-actor-${EXPERIMENT_NAME}`;
    this.criticPath =
      `indexeddb://flappy-ppo-critic-${EXPERIMENT_NAME}`;
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains(GAME_DATA_STORE)) {
          db.createObjectStore(GAME_DATA_STORE);
        }
      };
    });
  }

  async saveActor(model) {
    try {
      await model.save(this.actorPath);
      console.log('PPO Actor model salvo no IndexedDB');
    } catch (error) {
      console.error('Erro ao salvar PPO actor:', error);
    }
  }

  async loadActor() {
    try {
      const model = await tf.loadLayersModel(this.actorPath);
      console.log('PPO Actor model carregado do IndexedDB');
      return model;
    } catch (error) {
      console.log('Nenhum PPO actor model encontrado');
      return null;
    }
  }

  async saveCritic(model) {
    try {
      await model.save(this.criticPath);
      console.log('PPO Critic model salvo no IndexedDB');
    } catch (error) {
      console.error('Erro ao salvar PPO critic:', error);
    }
  }

  async loadCritic() {
    try {
      const model = await tf.loadLayersModel(this.criticPath);
      console.log('PPO Critic model carregado do IndexedDB');
      return model;
    } catch (error) {
      console.log('Nenhum PPO critic model encontrado');
      return null;
    }
  }

  async deleteActor() {
    try {
      await tf.io.removeModel(this.actorPath);
    } catch (error) {
      console.warn('PPO Actor model inexistente ou já removido');
    }
  }

  async deleteCritic() {
    try {
      await tf.io.removeModel(this.criticPath);
    } catch (error) {
      console.warn('PPO Critic model inexistente ou já removido');
    }
  }

  async saveGameData(key, data) {
    const db = await this.openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(GAME_DATA_STORE, 'readwrite');
      const store = transaction.objectStore(GAME_DATA_STORE);

      transaction.oncomplete = () => {
        db.close();
        resolve();
      };

      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };

      store.put(data, key);
    });
  }

  async loadGameData(key) {
    const db = await this.openDB();

    return new Promise((resolve) => {
      const transaction = db.transaction(GAME_DATA_STORE, 'readonly');
      const store = transaction.objectStore(GAME_DATA_STORE);
      const request = store.get(key);

      request.onsuccess = () => {
        db.close();
        resolve(request.result ?? null);
      };

      request.onerror = () => {
        db.close();
        resolve(null);
      };
    });
  }

  async deleteGameData(key) {
    const db = await this.openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(GAME_DATA_STORE, 'readwrite');
      const store = transaction.objectStore(GAME_DATA_STORE);

      transaction.oncomplete = () => {
        db.close();
        resolve();
      };

      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };

      store.delete(key);
    });
  }

  async saveMetadata(metadata) {
    await this.saveGameData('metadata', metadata);
  }

  async loadMetadata() {
    return this.loadGameData('metadata');
  }

  async clearAll() {
    await this.deleteActor();
    await this.deleteCritic();

    try {
      await this.deleteGameData('metadata');
    } catch (error) {
      console.warn('Não foi possível remover metadata:', error);
    }

    console.log('Memória do PPO resetada');
  }
}

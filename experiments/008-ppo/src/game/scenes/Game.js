import Phaser from 'phaser';
import {
	ACTIONS, ACTION_FLAP, ACTION_IDLE,
	PPOAgent, resetBrain
} from '../../rl/ppo.js';
import { recordEpisode } from '../../rl/episodeRecorder.js';

// Escala aplicada a TODAS as recompensas (mantém as proporções da referência).
// Com γ=0.99 e ~+1/frame, os retornos chegam perto de 100 e o critic não acompanha.
// Para copiar exatamente a referência, use 1.
const REWARD_SCALE = 0.1;

export class Game extends Phaser.Scene {
	constructor() {
		super('Game');
		this.gameOver = false;
		this.generation = 1;
		this.highScore = 0;
		this.lastState = null;
		this.lastAction = null;
		this.lastLogProb = null;
		this.lastValue = null;

		this.agent = new PPOAgent();
	}

	preload() {
		this.load.setPath('assets/sprites');
		this.load.image('bg', 'background-day.png');
		this.load.image('bird_up', 'bluebird-upflap.png');
		this.load.image('bird_mid', 'bluebird-midflap.png');
		this.load.image('bird_down', 'bluebird-downflap.png');
		this.load.image('pipeGreen', 'pipe-green.png');
		this.load.image('pipeRed', 'pipe-red.png');
		this.load.image('base', 'base.png');
	}

	async create() {

		this.ready = false;

		if (this.generation === 1) {
			const loaded = await this.agent.loadBrain();
			if (loaded.success) {
				this.generation = loaded.generation;
				this.highScore = loaded.highScore ?? 0;
			}
		}

		this.gameOver = false;
		this.score = 0;
		this.lastState = null;
		this.lastAction = null;
		this.lastLogProb = null;
		this.lastValue = null;

		this.zones = [];
		this.bonusReward = 0;
		this.proximityReward = 0;

		const bg = this.add.image(this.scale.width / 2, this.scale.height / 2, 'bg');
		bg.setDisplaySize(this.scale.width, this.scale.height);

		this.GROUND_H = 92;

		this.ground = this.add.tileSprite(0, this.scale.height, this.scale.width * 2, this.GROUND_H, 'base').setOrigin(0, 1);
		this.ground.setDepth(5);
		this.physics.add.existing(this.ground);
		this.ground.body.setAllowGravity(false);
		this.ground.body.setImmovable(true);

		this.bird = this.physics.add.sprite(120, this.scale.height / 2, 'bird_mid');
		this.bird.setGravityY(1000);

		if (!this.anims.exists('fly')) {
			this.anims.create({
				key: 'fly',
				frames: [
					{ key: 'bird_up' },
					{ key: 'bird_mid' },
					{ key: 'bird_down' }
				],
				frameRate: 10,
				repeat: -1
			});
		}

		this.bird.play('fly');
		this.bird.setDisplaySize(68, 48);
		this.bird.setBodySize(this.bird.width - 10, this.bird.height - 10);

		this.pipes = this.physics.add.group();

		this.scoreText = this.add.text(16, 16, 'Score: 0', {
			fontSize: '32px',
			fill: '#fff',
			stroke: '#000',
			strokeThickness: 4,
		});

		this.hudText = this.add.text(this.scale.width, 30, '', {
			fontSize: '18px',
			fill: '#ff0',
			stroke: '#000',
			strokeThickness: 3,
			align: 'left',
			fixedWidth: 220
		}).setOrigin(1, 0);

		this.hudText.setDepth(1000);

		this.hudBackground = this.add.rectangle(
			this.scale.width - 20,
			20,
			210,
			440,
			0x000000,
			0.2
		).setOrigin(1, 0);

		this.hudBackground.setStrokeStyle(1, 0x000000, 0.8);
		this.hudBackground.setDepth(999);

		this.scoreText.setDepth(1000);

		this.resetBtn = this.add.dom(
			this.scale.width - 43,
			20,
			'button',
			'width: 70px; height: 26px; background-color: #c0392b; color: #fff; ' +
			'border: none; border-radius: 8px; font-size: 12px; font-weight: normal; ' +
			'cursor: pointer; display: flex; align-items: center; justify-content: center;',
			'RESET'
		).setOrigin(0.5, 0.5);

		this.resetBtn.setDepth(1001);
		this.resetBtn.addListener('click');
		this.resetBtn.on('click', () => this.handleReset());

		this.lastPipeX = null;
		this.addPipeRow();

		this.physics.add.collider(this.bird, this.pipes, this.hitPipe, null, this);
		this.physics.add.collider(this.bird, this.ground, this.hitPipe, null, this);
		this.ready = true;
	}

	async update(time, delta) {
		if (this.gameOver || !this.ready) return;

		const pipeSpeed = Math.min(200 + this.score * 0.4, 400);

		this.pipes.setVelocityX(-pipeSpeed);

		this.zones.forEach(zone => {
			if (zone.body) {
				zone.body.setVelocityX(-pipeSpeed);
			}
		});

		const spawnDistance = 380 + (pipeSpeed - 200) * 0.7;

		if (this.lastPipeX === null) {
			this.lastPipeX = this.scale.width + 50;
		}

		const lastPipe = this.zones[this.zones.length - 1];

		if (lastPipe && this.lastPipeX - lastPipe.x >= spawnDistance) {
			this.lastPipeX = this.scale.width + 50;
			this.addPipeRow();
		}

		if (this.ground) {
			this.ground.tilePositionX += pipeSpeed * (delta / 1000);
		}

		// 1. Observar Estado

		const closestPipe = this.getClosestPipe();

		const pipesAhead = this.zones.filter(zone => zone.active && zone.x > this.bird.x)
			.sort((a, b) => a.x - b.x);

		let dx = 1058, dy = 0, velY = this.bird.body.velocity.y, gapHeight = 300,
			dxNext = 1058, dyNext = 0, gapNext = 300;

		if (pipesAhead.length > 0) {
			const current = pipesAhead[0];

			dx = current.x - this.bird.x;
			dx = Math.max(0, dx);

			dy = this.bird.y - current.body.center.y;
			gapHeight = current.height;

			if (pipesAhead.length > 1) {
				const next = pipesAhead[1];

				dxNext = next.x - this.bird.x;
				dxNext = Math.max(0, dxNext);

				dyNext = this.bird.y - next.body.center.y;
				gapNext = next.height;
			}
		} else {
			dy = this.bird.y - this.scale.height / 2;
			dyNext = this.bird.y - this.scale.height / 2;
		}

		const normalizedSpeed = ((pipeSpeed - 200) / 100) - 1;
		const normalizedDx = (dx / 529) - 1;
		const normalizedDxNext = (dxNext / 529) - 1;
		const normalizedGap = ((gapHeight - 200) / 105) - 1;
		const normalizedGapNext = ((gapNext - 200) / 105) - 1;
		const normalizedDy = Math.max(-1, Math.min(1, dy / 400));
		const normalizedDyNext = Math.max(-1, Math.min(1, dyNext / 400));
		const normalizedVelY = Math.max(-1, Math.min(1, velY / 1000));

		const currentState = [
			normalizedDx,
			normalizedDy,
			normalizedVelY,
			normalizedGap,
			normalizedDxNext,
			normalizedDyNext,
			normalizedGapNext,
			normalizedSpeed
		];

		// 2. Calcular Recompensa (mesma da referência REINFORCE)

		// Penalidade leve por velocidade vertical extrema
		const velPenalty = Math.abs(velY) > 700 ? -0.05 : 0;

		// Custo por flap
		const flapPenalty = this.lastAction === ACTION_FLAP ? -0.1 : 0;

		// Alinhamento com o centro do gap (gaussiana, 0 a +1 por frame)
		if (closestPipe) {
			const maxConsideredDy = this.scale.height / 2;
			const dyRatio = Math.abs(dy) / maxConsideredDy;
			const sigma = 0.5;

			this.proximityReward = Math.exp(-Math.pow(dyRatio, 2) / (2 * sigma * sigma));
		} else {
			this.proximityReward = 0;
		}

		// 3. Evaluate the CURRENT state before any possible PPO update.
		// This value is the correct bootstrap V(s_t) for the previous transition.

		const currentValue = this.agent.getValue(currentState);

		// 4. PPO: close the previous transition.
		//
		// The transition stored on the previous frame is:
		// (lastState, lastAction, reward, currentState)
		//
		// Therefore currentValue = V(currentState) is the correct bootstrap
		// value if this transition becomes the end of a rollout.

		if (this.lastState !== null && this.lastAction !== null) {
			const reward = (this.proximityReward + velPenalty + flapPenalty + this.bonusReward) * REWARD_SCALE;

			this.agent.collectStep(
				this.lastState,
				this.lastAction,
				reward,
				this.lastLogProb,
				this.lastValue,
				false,
				currentValue
			);
		}

		this.bonusReward = 0;

		// 5. Choose the next action AFTER a possible PPO update.

		const policyDecision = this.agent.chooseAction(currentState);
		const action = policyDecision.action;

		// 6. Executar Ação

		let actionStr = "IDLE";

		if (action === ACTION_FLAP) {
			actionStr = "FLAP";
			this.flap();
		}

		// 7. Armazenar para Próximo Frame

		this.lastState = currentState;
		this.lastAction = action;
		this.lastLogProb = policyDecision.logProb;
		this.lastValue = policyDecision.value;

		// 8. Física e Limpeza

		if (this.bird.angle < 20) {
			this.bird.angle += 1;
		}

		this.pipes.getChildren().forEach((pipe) => {
			const w = (pipe.displayWidth !== undefined) ? pipe.displayWidth : (pipe.width || 0);

			if (pipe.x + w < 0) {
				pipe.destroy();
			}
		});

		if (this.zones) {
			for (let i = this.zones.length - 1; i >= 0; i--) {
				const zone = this.zones[i];

				if (zone.x + zone.width < 0) {
					zone.destroy();
					this.zones.splice(i, 1);
				}
			}
		}

		if (this.bird.y > this.scale.height + 50 || this.bird.y < -50) {
			this.hitPipe();
		}

		// 9. Atualizar HUD

		const policy = this.agent.getPolicy(currentState);
		const buffer = this.agent.buffer;

		this.hudText.setText(
			`Gen: ${this.generation}\n` +
			`High: ${this.highScore}\n` +
			`Speed: ${Math.floor(pipeSpeed)}\n` +
			`DX: ${Math.floor(dx)}\n` +
			`DY: ${Math.floor(dy)}\n` +
			`VelY: ${Math.floor(velY)}\n` +
			`Gap: ${Math.floor(gapHeight)}\n` +
			`DXNext: ${Math.floor(dxNext)}\n` +
			`DYNext: ${Math.floor(dyNext)}\n` +
			`GapNext: ${Math.floor(gapNext)}\n` +
			`Prox: ${this.proximityReward.toFixed(2)}\n` +
			`P-Idle: ${(policy[ACTION_IDLE] * 100).toFixed(1)}%\n` +
			`P-Flap: ${(policy[ACTION_FLAP] * 100).toFixed(1)}%\n` +
			`V(s): ${this.lastValue != null ? this.lastValue.toFixed(2) : '-'}\n` +
			`Buffer: ${buffer.size}/128\n` +
			`A-Loss: ${this.agent.lastActorLoss === null ? '-' : this.agent.lastActorLoss.toFixed(4)}\n` +
			`C-Loss: ${this.agent.lastCriticLoss === null ? '-' : this.agent.lastCriticLoss.toFixed(4)}\n` +
			`Clip%: ${this.agent.lastClipFraction === null ? '-' : (this.agent.lastClipFraction * 100).toFixed(1)}%\n` +
			`Action: ${actionStr}`
		);
	}

	flap() {
		if (this.gameOver) return;

		this.bird.setVelocityY(-350);
		this.bird.angle = -20;
	}

	async handleReset() {
		const confirmed = confirm('Apagar toda a memória e os dados gravados no navegador? Esta ação não pode ser desfeita.');

		if (!confirmed) return;

		await resetBrain();
		window.location.reload();
	}

	getClosestPipe() {
		let closest = null;
		let minDist = Infinity;

		if (!this.zones) return null;

		this.zones.forEach(zone => {
			if (zone.active && zone.x > this.bird.x) {
				const dist = zone.x - this.bird.x;

				if (dist < minDist) {
					minDist = dist;
					closest = zone;
				}
			}
		});

		return closest;
	}

	addPipeRow() {
		const gap = Phaser.Math.Between(200, 410);
		let centerY = Phaser.Math.Between(150, this.scale.height - 150);
		const x = this.scale.width + 50;

		const bottomMouthLimit = this.scale.height - this.GROUND_H;

		if (centerY + gap / 2 > bottomMouthLimit) {
			centerY = bottomMouthLimit - gap / 2;
		}

		const pipeKey = Phaser.Math.Between(0, 1) === 0 ? 'pipeGreen' : 'pipeRed';

		const top = this.pipes.create(x, centerY - gap / 2, pipeKey).setOrigin(0, 1);
		top.body.allowGravity = false;
		top.setImmovable(true);
		top.setVelocityX(-200);
		top.setFlipY(true);
		top.setDisplaySize(104, 640);

		const topMouthBottomY = centerY - gap / 2;
		const topMouthHeight = top.displayHeight;
		const topMouthTopY = topMouthBottomY - topMouthHeight;
		const topBodyHeight = Math.max(0, Math.floor(topMouthTopY));

		if (topBodyHeight > 0) {
			const topBody = this.add.rectangle(
				x + top.displayWidth / 2,
				topMouthTopY / 2,
				top.displayWidth,
				topBodyHeight
			);

			topBody.setOrigin(0.5, 0.5);
			this.physics.add.existing(topBody);
			topBody.body.setAllowGravity(false);
			topBody.body.setImmovable(true);
			topBody.body.setVelocityX(-200);
			topBody.setVisible(false);
			this.pipes.add(topBody);
		}

		const bottom = this.pipes.create(x, centerY + gap / 2, pipeKey).setOrigin(0, 0);
		bottom.body.allowGravity = false;
		bottom.setImmovable(true);
		bottom.setVelocityX(-200);
		bottom.setDisplaySize(104, 640);

		const bottomMouthTopY = centerY + gap / 2;
		const bottomMouthHeight = bottom.displayHeight;
		const startY = bottomMouthTopY + bottomMouthHeight;
		const bottomBodyHeight = Math.max(0, this.scale.height - startY);

		if (bottomBodyHeight > 0) {
			const bottomBodyCenterY = startY + bottomBodyHeight / 2;

			const bottomBody = this.add.rectangle(
				x + bottom.displayWidth / 2,
				bottomBodyCenterY,
				bottom.displayWidth,
				bottomBodyHeight
			);

			bottomBody.setOrigin(0.5, 0.5);
			this.physics.add.existing(bottomBody);
			bottomBody.body.setAllowGravity(false);
			bottomBody.body.setImmovable(true);
			bottomBody.body.setVelocityX(-200);
			bottomBody.setVisible(false);
			this.pipes.add(bottomBody);
		}

		const zone = this.add.zone(x + 104, centerY, 2, gap);

		this.physics.world.enable(zone);
		zone.body.setVelocityX(-200);
		zone.body.allowGravity = false;
		zone.scored = false;
		zone.active = true;

		if (!this.zones) this.zones = [];

		this.zones.push(zone);

		this.physics.add.overlap(this.bird, zone, (bird, z) => {
			if (!z.scored) {
				z.scored = true;
				this.score++;
				this.scoreText.setText('Score: ' + this.score);
				this.bonusReward = 10;
			}
		});
	}

	async hitPipe() {
		if (this.gameOver) return;

		this.gameOver = true;

		const deathReward = -20 * REWARD_SCALE;

		if (this.lastState !== null && this.lastAction !== null) {
			// Terminal step: done=true, V(s')=0
			// The value is 0 for terminal steps because there is no next state.

			const stored = this.agent.buffer.add(
				this.lastState,
				this.lastAction,
				deathReward,
				this.lastValue,
				this.lastLogProb,
				true
			);

			// Only run a PPO update if the terminal step was stored and filled
			// the buffer. If it was dropped (buffer still full while training
			// runs), the next episode's first collectStep() triggers training
			// with a correct bootstrap value instead of forcing 0 here.
			// Otherwise keep accumulating across episode boundaries so the
			// buffer fills naturally and batches never become tiny.
			if (stored && this.agent.buffer.isReady()) {
				await this.agent.forceUpdate(0);
			}
		}

		this.highScore = Math.max(this.highScore, this.score);
		recordEpisode(this.generation, this.score);

		this.generation++;

		await this.agent.saveBrain(this.generation, this.highScore);

		this.endGame();
	}

	endGame() {
		if (this.pipes) {
			try { this.pipes.setVelocityX(0); } catch (e) { }
		}

		if (this.zones) {
			this.zones.forEach(zone => {
				if (zone.body) zone.body.setVelocityX(0);
			});
		}

		if (this.bird && this.bird.body) {
			try {
				this.bird.setVelocity(0);
				this.bird.body.setAllowGravity(false);
			} catch (e) { }
		}

		try { this.physics.pause(); } catch (e) { }
		try { this.anims.pauseAll(); } catch (e) { }

		this.add
			.text(
				this.scale.width / 2,
				this.scale.height / 2 - 40,
				'Game Over',
				{
					fontSize: '64px',
					fill: '#fff',
					stroke: '#000',
					strokeThickness: 6,
				}
			)
			.setOrigin(0.5)
			.setDepth(1000);

		this.time.delayedCall(500, () => {
			this.anims.resumeAll();
			this.physics.resume();
			this.scene.restart();
		});
	}
}
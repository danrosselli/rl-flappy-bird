import Phaser from 'phaser';
import {
	ACTIONS, ACTION_FLAP, ACTION_IDLE,
	ActorCriticAgent, resetBrain
} from '../../rl/policyGradient.js';
import { recordEpisode } from '../../rl/episodeRecorder.js';

export class Game extends Phaser.Scene {
	constructor() {
		super('Game');
		this.gameOver = false;
		this.generation = 1;
		this.highScore = 0;
		this.lastState = null;
		this.lastAction = null;
		this.lastValue = null;
		this.agent = new ActorCriticAgent();
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

		// ← NOVA FLAG
		this.ready = false; // Indica que o create() terminou

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

		// HUD Text (agora no lado direito)
		this.hudText = this.add.text(this.scale.width, 30, '', {
			fontSize: '18px',
			fill: '#ff0',
			stroke: '#000',
			strokeThickness: 3,
			align: 'left',
			fixedWidth: 220
		}).setOrigin(1, 0);  // Origem no canto superior direito
		this.hudText.setDepth(1000);

		// Moldura (fundo semi-transparente preto com borda)
		this.hudBackground = this.add.rectangle(
			this.scale.width - 20,  // Mesmo X do texto (alinhado à direita)
			20,                     // Mesmo Y do topo do texto
			210,                    // Largura fixa (ajuste se precisar mais/menos)
			400,                    // Altura aproximada (cobre todo o texto)
			0x000000,               // Cor preta
			0.2                     // Alpha = semi-transparente
		).setOrigin(1, 0);          // Origem no canto superior direito

		this.hudBackground.setStrokeStyle(1, 0x000000, 0.8);  // Borda preta opcional
		this.hudBackground.setDepth(999);  // Logo atrás do texto

		this.scoreText.setDepth(1000);

		// Botão de reset nativo (DOMElement = botão HTML real que escala com o canvas)
		this.resetBtn = this.add.dom(
			this.scale.width - 43,  // X (centro do botão, canto superior direito)
			20,                     // Y (centro do botão)
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
		if (this.gameOver || !this.ready) return; // ← PROTEÇÃO AQUI

		// NOVA: Acelera pipes com score + CAP MÁXIMO 450 (evita impossível)
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
			.sort((a, b) => a.x - b.x);  // Ordena por proximidade (trailing edge)

		let dx = 1058, dy = 0, velY = this.bird.body.velocity.y, gapHeight = 300,
			dxNext = 1058, dyNext = 0, gapNext = 300;

		if (pipesAhead.length > 0) {
			const current = pipesAhead[0];
			dx = current.x - this.bird.x;
			dx = Math.max(0, dx);  // Clipa para 0 após passar o trailing edge
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

		// Normaliza velocidade de 200 a 400 para o intervalo [-1, 1]
		// 200 -> -1.0 | 300 -> 0.0 | 400 -> +1.0
		const normalizedSpeed = ((pipeSpeed - 200) / 100) - 1;

		// Normaliza dx (que no seu código vai de 0 a 1058) para o intervalo [-1, 1]
		// 0 -> -1.0 | 529 -> 0.0 | 1058 -> +1.0
		const normalizedDx = (dx / 529) - 1;
		const normalizedDxNext = (dxNext / 529) - 1;

		// Gaps (200 a 410) para [-1, 1]
		const normalizedGap = ((gapHeight - 200) / 105) - 1;
		const normalizedGapNext = ((gapNext - 200) / 105) - 1;

		// dy e velY já oscilam entre negativo e positivo, só ajuste o limite
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

		// 2. Calcular Recompensa de Proximidade (nova versão mais estável)

		// Penalidade leve por velocidade vertical extrema (evita loops infinitos de flap)
		const velPenalty = Math.abs(velY) > 700 ? -0.05 : 0;

		// Custo por flap: penalidade para evitar espamar flap
		const flapPenalty = this.lastAction === ACTION_FLAP ? -0.05 : 0;

		// Recompensa de alinhamento com o gap (só se houver pipe próximo)
		if (closestPipe) {
			const gap = closestPipe.height;
			const halfGap = gap / 2;
			const absDy = Math.abs(dy);

			// Dinâmico: cobre metade da tela + margem pequena
			const maxConsideredDy = this.scale.height / 2;  // 384 + 50 = 434 (boa margem)

			// Normalização ampla: 0 = centro, 1 = na distância máxima considerada
			const normalizedDy = absDy / maxConsideredDy;

			// Gaussiana mais larga (sigma maior) sem offset
			// sigma = 0.8 dá uma curva que cobre bem a tela
			const sigma = 0.5;
			const gaussian = Math.exp(-Math.pow(normalizedDy, 2) / (2 * sigma * sigma));

			this.proximityReward = gaussian;
		} else {
			this.proximityReward = 0;
		}

		// 3. Online Actor-Critic update (one-step TD).
		//    No trajectory buffer — O(1) memory per episode.
		if (this.lastState !== null && this.lastAction !== null) {
			const reward = this.proximityReward + velPenalty + flapPenalty + this.bonusReward;
			this.agent.update(
				this.lastState, this.lastAction, this.lastValue,
				reward, currentState, false
			);
		}
		this.bonusReward = 0;

		// 4. Escolher a ação pela política probabilística.
		const policyDecision = this.agent.chooseAction(currentState);
		const action = policyDecision.action;

		// 5. Executar Ação
		let actionStr = "IDLE";
		if (action === ACTION_FLAP) {
			actionStr = "FLAP";
			this.flap();
		}

		// 6. Armazenar para Próximo Frame
		this.lastState = currentState;
		this.lastAction = action;
		this.lastValue = policyDecision.value;

		// 7. Física e Limpeza
		if (this.bird.angle < 20) {
			this.bird.angle += 1;
		}

		this.pipes.getChildren().forEach((pipe) => {
			const w = (pipe.displayWidth !== undefined) ? pipe.displayWidth : (pipe.width || 0);
			if (pipe.x + w < 0) pipe.destroy();
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

		// 8. Atualizar HUD com as probabilidades da política.
		const policy = this.agent.getPolicy(currentState);
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
			`A-Loss: ${this.agent.lastTrainingLoss === null ? '-' : this.agent.lastTrainingLoss.toFixed(4)}\n` +
			`C-Loss: ${this.agent.lastCriticLoss === null ? '-' : this.agent.lastCriticLoss.toFixed(4)}\n` +
			`Action: ${actionStr}`
		);

	}

	flap() {
		if (this.gameOver) return;
		this.bird.setVelocityY(-350);
		this.bird.angle = -20;
	}

	// Reset completo: apaga toda a memória (modelo + buffers + metadata) e recarga a página
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
		top.setVelocityX(-200);  // Inicial, update() corrige
		top.setFlipY(true);
		top.setDisplaySize(104, 640);

		const topMouthBottomY = centerY - gap / 2;
		const topMouthHeight = top.displayHeight;
		const topMouthTopY = topMouthBottomY - topMouthHeight;
		const topBodyHeight = Math.max(0, Math.floor(topMouthTopY));
		if (topBodyHeight > 0) {
			const topBody = this.add.rectangle(x + top.displayWidth / 2, topMouthTopY / 2, top.displayWidth, topBodyHeight);
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
		bottom.setVelocityX(-200);  // Inicial, update() corrige
		bottom.setDisplaySize(104, 640);

		const bottomMouthTopY = centerY + gap / 2;
		const bottomMouthHeight = bottom.displayHeight;
		const startY = bottomMouthTopY + bottomMouthHeight;
		const bottomBodyHeight = Math.max(0, this.scale.height - startY);
		if (bottomBodyHeight > 0) {
			const bottomBodyCenterY = startY + bottomBodyHeight / 2;
			const bottomBody = this.add.rectangle(x + bottom.displayWidth / 2, bottomBodyCenterY, bottom.displayWidth, bottomBodyHeight);
			bottomBody.setOrigin(0.5, 0.5);
			this.physics.add.existing(bottomBody);
			bottomBody.body.setAllowGravity(false);
			bottomBody.body.setImmovable(true);
			bottomBody.body.setVelocityX(-200);
			bottomBody.setVisible(false);
			this.pipes.add(bottomBody);
		}

		// Zone movida para o final do cano (borda trailing / direita, x + 104)
		const zone = this.add.zone(x + 104, centerY, 2, gap);
		this.physics.world.enable(zone);
		zone.body.setVelocityX(-200);  // Inicial, update() corrige
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

		const deathReward = -20;
		if (this.lastState !== null && this.lastAction !== null) {
			// Terminal update: done=true forces V(s') = 0 in the advantage calculation.
			this.agent.update(
				this.lastState, this.lastAction, this.lastValue,
				deathReward, this.lastState, true
			);
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
		// NOVA: Para zones também
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
			.text(this.scale.width / 2, this.scale.height / 2 - 40, 'Game Over', {
				fontSize: '64px',
				fill: '#fff',
				stroke: '#000',
				strokeThickness: 6,
			})
			.setOrigin(0.5)
			.setDepth(1000);

		this.time.delayedCall(500, () => {
			this.anims.resumeAll();
			this.physics.resume();
			this.scene.restart();
		});
	}
}
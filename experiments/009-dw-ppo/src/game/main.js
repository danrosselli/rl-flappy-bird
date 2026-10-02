import { Game as MainGame } from './scenes/Game';
import { AUTO, Scale, Game } from 'phaser';

const config = {
	type: AUTO,
	width: 1024,
	height: 768,
	parent: 'game-container',
	backgroundColor: '#028af8',
	dom: {
		createContainer: true
	},
	physics: {
		default: 'arcade',
		arcade: {
			gravity: { y: 0 },
			debug: false
		}
	},
	scale: {
		mode: Scale.FIT,
		autoCenter: Scale.CENTER_BOTH
	},
	scene: [
		MainGame
	]
};

const StartGame = (parent) => {
	return new Game({ ...config, parent });
}

export default StartGame;

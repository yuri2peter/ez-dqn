import { DQNAgent } from "../src";
import { BOARD_WIDTH, State } from "./defines";
import { paint } from "./paint";
import { save } from "./persist";

const MOVE_SPPED = 0.02;
const TICK_MS = 0;
// const TICK_MS = 50;

export class Game {
  tickIndex = 0;
  state: State = {
    goldX: 0.2,
    goldY: 0.2,
    bombX: 0.8,
    bombY: 0,
    agentX: 0.5,
  };
  goldNum = 0;
  bombNum = 0;
  reward = 0;
  tderror = 0;
  agent = new DQNAgent({
    numStates: 5, // 自身位置, 金币位置，炸弹位置
    maxNumActions: 3, // 向左，保持，向右
    num_hidden_units: 64,
    opt: {
      alpha: 0.02,
      learning_steps_per_iteration: 20,
      epsilon: 0.2,
      gamma: 0.8,
    },
  });

  run() {
    setInterval(() => {
      this.tick();
    }, TICK_MS);
  }

  tick() {
    this.tickIndex++;
    if (this.tickIndex % 10000 === 0) {
      // save game
      save(this);
    }
    const { goldX, goldY, bombX, bombY, agentX } = this.state;
    const act = this.agent.act([goldX, goldY, bombX, bombY, agentX]);
    let reward = 0;
    switch (act) {
      // left
      case 0:
        if (agentX < MOVE_SPPED) {
          this.state.agentX = 0;
          reward = -0.2;
        } else {
          this.state.agentX -= MOVE_SPPED;
        }
        break;
      // stay
      case 1:
        reward = 0.0;
        break;
      // right
      case 2:
        if (agentX > 1 - MOVE_SPPED) {
          this.state.agentX = 1;
          reward = -0.2;
        } else {
          this.state.agentX += MOVE_SPPED;
        }
        break;
    }
    // drop
    this.state.goldY += MOVE_SPPED;
    this.state.bombY += MOVE_SPPED;
    if (this.state.goldY >= 1) {
      if (Math.abs(this.state.agentX - this.state.goldX) <= BOARD_WIDTH / 200) {
        reward = 0.6;
        this.goldNum++;
      }
      this.state.goldX = Math.random();
      this.state.goldY = Math.random() / 4;
    }
    if (this.state.bombY >= 1) {
      if (Math.abs(this.state.agentX - this.state.bombX) <= BOARD_WIDTH / 200) {
        reward = -1;
        this.bombNum++;
      }
      this.state.bombX = Math.random();
      this.state.bombY = Math.random() / 4;
    }
    const tderror = this.agent.learn(reward);
    this.reward = this.reward * 0.99 + reward * 0.01;
    this.tderror = this.tderror * 0.99 + tderror * 0.01;
    paint(this);
  }
}

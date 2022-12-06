import { DQNAgent } from "../src";
import { BOARD_WIDTH, State } from "./defines";
import { paint } from "./paint";

const MOVE_SPPED = 0.02;

export class Game {
  tickIndex = 0;
  state: State = {
    goldX: 0.2,
    goldY: 0.2,
    bombX: 0.8,
    bombY: 0,
    agentX: 0.5,
  };

  agent = new DQNAgent({
    numStates: 5, // 自身位置, 金币位置，炸弹位置
    maxNumActions: 3, // 向左，保持，向右
    num_hidden_units: 100,
  });

  run() {
    setInterval(() => {
      this.tick();
    }, 100);
  }

  tick() {
    this.tickIndex++;
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
        reward = 0;
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
        reward = 0.5;
      }
      this.state.goldX = Math.random();
      this.state.goldY = Math.random() / 4;
    }
    if (this.state.bombY >= 1) {
      if (Math.abs(this.state.agentX - this.state.bombX) <= BOARD_WIDTH / 200) {
        reward = -1;
      }
      this.state.bombX = Math.random();
      this.state.bombY = Math.random() / 4;
    }
    this.agent.learn(reward);
    paint(this.state);
  }
}

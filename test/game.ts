import { DQNAgent } from "../src";
import { State } from "./defines";
import { paint } from "./paint";

export class Game {
  state: State = {
    goldX: 0.2,
    goldY: 0.2,
    bombX: 0.8,
    bombY: 0,
    agentX: 0.5,
  };

  agent = new DQNAgent({
    numStates: 3, // 自身位置, 金币位置，炸弹位置
    maxNumActions: 3, // 向左，保持，向右
    num_hidden_units: 100,
  });

  paintState() {
    paint(this.state);
  }
}

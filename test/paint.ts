import { BOARD_WIDTH, State } from "./defines";
import blessed from "blessed";

const screen = blessed.screen({
  smartCSR: true,
});
screen.title = "EZDQN Test";
const box = blessed.box({
  top: "center",
  left: "center",
  width: "50%",
  height: "100%",
  border: {
    type: "line",
  },
  style: {
    fg: "white",
    border: {
      fg: "#f0f0f0",
    },
  },
});

const agentBox = blessed.line({
  bottom: 0,
  left: "center",
  orientation: "horizontal",
  width: BOARD_WIDTH + "%",
  style: {
    bg: "blue",
  },
});

const goldBox = blessed.box({
  top: "0%",
  left: "0%",
  width: 1,
  height: 1,
  style: {
    bg: "yellow",
  },
});

const bombBox = blessed.box({
  top: "0%",
  left: "0%",
  width: 1,
  height: 1,
  style: {
    bg: "grey",
  },
});

screen.append(box);
box.append(agentBox);
box.append(goldBox);
box.append(bombBox);

export function paint(state: State) {
  agentBox.left = state.agentX * 100 - BOARD_WIDTH / 2 + "%";
  goldBox.left = state.goldX * 100 + "%";
  goldBox.top = state.goldY * 100 + "%";
  bombBox.left = state.bombX * 100 + "%";
  bombBox.top = state.bombY * 100 + "%";
  screen.render();
}

screen.render();
screen.key(["escape", "q", "C-c"], function (ch, key) {
  return process.exit(0);
});

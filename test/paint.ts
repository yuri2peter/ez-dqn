import { State } from "./defines";
import readline from "readline";
const outStream = process.stdout;
const rl = readline.createInterface({
  input: process.stdin,
  output: outStream,
});

const isTTy = process.stdout.isTTY;
if (!isTTy) {
  console.log("当前控制台不支持TTY，无法显示");
}

export function paint(state: State) {
  if (!isTTy) {
    return;
  }
  readline.cursorTo(outStream, 0, 0);
  readline.clearScreenDown(outStream);

  paintBox();
  paintObj(state);
  writeAt(0, 10, "");
}

function paintBox() {
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
      if (i === 0 && j === 0) {
        writeAt(i, j, "┌");
        continue;
      }
      if (i === 0 && j === 9) {
        writeAt(i, j, "└");
        continue;
      }
      if (i === 9 && j === 0) {
        writeAt(i, j, "┐");
        continue;
      }
      if (i === 9 && j === 9) {
        writeAt(i, j, "┘");
        continue;
      }
    }
  }
}

function paintObj(state: State) {
  const agentX = floatToPx(state.agentX);
  writeAt(agentX - 1, 9, "=");
  writeAt(agentX, 9, "=");
  writeAt(agentX + 1, 9, "=");

  writeAt(floatToPx(state.goldX), floatToPx(state.goldY), "O");
  writeAt(floatToPx(state.bombX), floatToPx(state.bombY), "X");
}

function writeAt(x: number, y: number, s: string) {
  readline.cursorTo(outStream, x, y);
  rl.write(s);
}

function floatToPx(f: number) {
  return Math.round(f * 10);
}

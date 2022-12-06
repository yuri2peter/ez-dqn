import fs from "fs";
import path from "path";
import { Game } from "./game";

const f1 = path.resolve(__dirname, "save1.json");
const f2 = path.resolve(__dirname, "save2.json");

export function save(game: Game) {
  const j = game.agent.toJSON();
  fs.writeFileSync(path.resolve(__dirname, "save2.json"), JSON.stringify(j));
}

export function load(game: Game) {
  if (fs.existsSync(f2)) {
    const c2 = fs.readFileSync(f2, "utf8");
    game.agent.fromJSON(JSON.parse(c2));
    return;
  }
  if (fs.existsSync(f1)) {
    const c1 = fs.readFileSync(f1, "utf8");
    game.agent.fromJSON(JSON.parse(c1));
  }
}

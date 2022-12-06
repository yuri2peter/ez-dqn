import { Game } from "./game";
import { load } from "./persist";

async function main() {
  const game = new Game();
  load(game);
  game.run();
}

main();

// nets are hardcoded for now as key (str) -> Mat
// not proud of this. better solution is to have a whole Net object
import { Mat, MatJson, RandMat, updateMat } from "./Mat";

type Keys = "W1" | "b1" | "W2" | "b2";
export type NetJson = {
  [T in Keys]: MatJson;
};
export class Net {
  W1: Mat;
  b1: Mat;
  W2: Mat;
  b2: Mat;
  constructor({ nh, ns, na }: { nh: number; ns: number; na: number }) {
    this.W1 = RandMat(nh, ns, 0, 0.01);
    this.b1 = new Mat(nh, 1);
    this.W2 = RandMat(na, nh, 0, 0.01);
    this.b2 = new Mat(na, 1);
  }
  forEach(cb: (v: Mat, k: Keys) => void) {
    cb(this.W1, "W1");
    cb(this.b1, "b1");
    cb(this.W2, "W2");
    cb(this.b2, "b2");
  }
}

export function updateNet(net: Net, alpha: number) {
  net.forEach((v) => {
    updateMat(v, alpha);
  });
}

export function netToJSON(net: Net) {
  const j = {};
  net.forEach((v, k) => {
    j[k] = v.toJSON();
  });
  return j as NetJson;
}

export function netFromJSON(j: NetJson) {
  const net = new Net({ nh: 0, ns: 0, na: 0 });
  net.forEach((v, k) => {
    net[k] = new Mat(1, 1); // not proud of this
    net[k].fromJSON(j[k]);
  });
  return net;
}

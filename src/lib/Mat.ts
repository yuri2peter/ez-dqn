import { zeros, assert, randn } from "./utils";

export type MatJson = { n: number; d: number; w: number[] };
export class Mat {
  n: number;
  d: number;
  w: number[];
  dw: number[];
  constructor(n: number, d: number) {
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  }
  get(row: number, col: number) {
    // slow but careful accessor function
    // we want row-major order
    const ix = this.d * row + col;
    assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  }
  set(row: number, col: number, v: number) {
    // slow but careful accessor function
    const ix = this.d * row + col;
    assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  }
  setFrom(arr: number[]) {
    for (let i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i];
    }
  }
  setColumn(m: { w: number[] }, i: number) {
    for (let q = 0, n = m.w.length; q < n; q++) {
      this.w[this.d * q + i] = m.w[q];
    }
  }
  toJSON(): MatJson {
    const { n, d, w } = this;
    return { n, d, w };
  }
  fromJSON(json: MatJson) {
    this.n = json.n;
    this.d = json.d;
    this.w = zeros(this.n * this.d);
    this.dw = zeros(this.n * this.d);
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
  }
}

export function updateMat(m: Mat, alpha: number) {
  // updates in place
  for (let i = 0, n = m.n * m.d; i < n; i++) {
    if (m.dw[i] !== 0) {
      m.w[i] += -alpha * m.dw[i];
      m.dw[i] = 0;
    }
  }
}

export function RandMat(n: number, d: number, mu: number, std: number) {
  let m = new Mat(n, d);
  fillRandn(m, mu, std);
  return m;
}

// Mat utils
// fill matrix with random gaussian numbers
function fillRandn(m: Mat, mu: number, std: number) {
  for (let i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randn(mu, std);
  }
}

// 填充常数
export function gradFillConst(m: Mat, c: number) {
  for (let i = 0, n = m.dw.length; i < n; i++) {
    m.dw[i] = c;
  }
}

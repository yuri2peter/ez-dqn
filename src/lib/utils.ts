export function assert(
  condition: Boolean,
  message: string = "Assertion failed"
) {
  // from http://stackoverflow.com/questions/15313418/javascript-assert
  if (!condition) {
    if (typeof Error !== "undefined") {
      throw new Error(message);
    }
    throw message; // Fallback
  }
}

// helper function returns array of zeros of length n
// and uses typed arrays if available
export function zeros(n: number) {
  return Array.from({ length: n }).map((t) => 0);
}

// Random numbers utils
const gaussRandom = ((): (() => number) => {
  let return_v = false;
  let v_val = 0.0;
  const a = () => {
    if (return_v) {
      return_v = false;
      return v_val;
    }
    const u = 2 * Math.random() - 1;
    const v = 2 * Math.random() - 1;
    const r = u * u + v * v;
    if (r == 0 || r > 1) return a();
    const c = Math.sqrt((-2 * Math.log(r)) / r);
    v_val = v * c; // cache this
    return_v = true;
    return u * c;
  };
  return a;
})();
export function randf(a: number, b: number) {
  return Math.random() * (b - a) + a;
}
export function randi(a: number, b: number) {
  return Math.floor(Math.random() * (b - a) + a);
}
export function randn(mu: number, std: number) {
  return mu + gaussRandom() * std;
}

export function sig(x: number) {
  // helper function for computing sigmoid
  return 1.0 / (1 + Math.exp(-x));
}

export function maxi(w: number[]) {
  // argmax of array w
  let maxv = w[0];
  let maxix = 0;
  for (let i = 1, n = w.length; i < n; i++) {
    let v = w[i];
    if (v > maxv) {
      maxix = i;
      maxv = v;
    }
  }
  return maxix;
}

export function samplei(w: number[]) {
  // sample argmax from w, assuming w are
  // probabilities that sum to one
  let r = randf(0, 1);
  let x = 0.0;
  let i = 0;
  while (true) {
    x += w[i];
    if (x > r) {
      return i;
    }
    i++;
  }
}

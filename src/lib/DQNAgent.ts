import { Graph } from "./Graph";
import { Mat } from "./Mat";
import { Net, netFromJSON, NetJson, netToJSON, updateNet } from "./Net";
import { randi, maxi } from "./utils";

interface Opt {
  gamma: number;
  epsilon: number;
  alpha: number;
  experience_add_every: number;
  experience_size: number;
  learning_steps_per_iteration: number;
  tderror_clamp: number;
}

type DQNAgentJson = {
  nh: number;
  ns: number;
  na: number;
  net: NetJson;
};

export class DQNAgent {
  private gamma: number = 0.75;
  private epsilon: number = 0.1;
  private alpha: number = 0.01;
  private experience_add_every: number = 25;
  private experience_size: number = 5000;
  private learning_steps_per_iteration: number = 10;
  private tderror_clamp: number = 1.0;
  private numStates!: number;
  private maxNumActions!: number;
  private num_hidden_units!: number;

  private nh!: number;
  private ns!: number;
  private na!: number;
  private net!: Net;
  // this.s0, this.a0, this.r0, this.s1, this.a1
  private exp!: [Mat, number, number, Mat, number][];
  private expi!: number;
  private t!: number;
  private r0!: number;
  private s0!: Mat;
  private s1!: Mat;
  private a0!: number;
  private a1!: number;
  private lastG!: Graph;

  constructor({
    opt = {},
    numStates,
    maxNumActions,
    num_hidden_units = 100,
  }: {
    opt?: Partial<Opt>;
    numStates: number;
    maxNumActions: number;
    num_hidden_units?: number;
  }) {
    Object.assign(this, opt, { numStates, maxNumActions, num_hidden_units });
    this.reset();
  }
  /**
   * 调参
   * @param opt
   */
  ajust(opt: Partial<Opt>) {
    Object.assign(this, opt);
  }

  /**
   * 重置agent
   */
  reset() {
    this.nh = this.num_hidden_units; // number of hidden units
    this.ns = this.numStates;
    this.na = this.maxNumActions;

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    this.net = new Net({
      nh: this.nh,
      ns: this.ns,
      na: this.na,
    });

    this.exp = []; // experience
    this.expi = 0; // where to insert

    this.t = 0;

    this.r0 = 0;
    this.s0 = new Mat(0, 0);
    this.s1 = new Mat(0, 0);
    this.a0 = 0;
    this.a1 = 0;
  }

  /**
   * 保存网络主体
   */
  toJSON(): DQNAgentJson {
    return {
      nh: this.nh,
      ns: this.ns,
      na: this.na,
      net: netToJSON(this.net),
    };
  }

  /**
   * 加载网络主体
   */
  fromJSON(j: DQNAgentJson) {
    // load function
    this.nh = j.nh;
    this.ns = j.ns;
    this.na = j.na;
    this.net = netFromJSON(j.net);
  }
  private forwardQ(net: Net, s: Mat, needs_backprop: boolean) {
    const g = new Graph(needs_backprop);
    const a1mat = g.add(g.mul(net.W1, s), net.b1);
    const h1mat = g.tanh(a1mat);
    const a2mat = g.add(g.mul(net.W2, h1mat), net.b2);
    this.lastG = g; // back this up. Kind of hacky isn't it
    return a2mat;
  }

  /**
   * 输出action
   * @param slist 环境
   * @returns action index
   */
  act(slist: number[]) {
    // convert to a Mat column vector
    const s = new Mat(this.ns, 1);
    s.setFrom(slist);
    let a = 0;
    // epsilon greedy policy
    if (Math.random() < this.epsilon) {
      a = randi(0, this.na);
    } else {
      // greedy wrt Q function
      const amat = this.forwardQ(this.net, s, false);
      a = maxi(amat.w); // returns index of argmax action
    }

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  }

  /**
   * 学习
   * @param r1 奖励
   * @returns tderror
   */
  learn(r1: number) {
    // perform an update on Q function
    // learn from this tuple to get a sense of how "surprising" it is to the agent
    const tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1);

    // decide if we should keep this experience in the replay
    if (this.t % this.experience_add_every === 0) {
      this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
      this.expi += 1;
      if (this.expi > this.experience_size) {
        this.expi = 0;
      } // roll over when we run out
    }
    this.t += 1;

    // sample some additional experience from replay memory and learn from it
    for (let k = 0; k < this.learning_steps_per_iteration; k++) {
      const ri = randi(0, this.exp.length); // todo: priority sweeps?
      const e = this.exp[ri];
      this.learnFromTuple(e[0], e[1], e[2], e[3]);
    }
    this.r0 = r1; // store for next update
    return tderror;
  }

  private learnFromTuple(s0: Mat, a0: number, r0: number, s1: Mat) {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')

    // compute the target Q value
    const tmat = this.forwardQ(this.net, s1, false);
    const qmax = r0 + this.gamma * tmat.w[maxi(tmat.w)];

    // now predict
    const pred = this.forwardQ(this.net, s0, true);

    let tderror = pred.w[a0] - qmax;
    const clamp = this.tderror_clamp;
    if (Math.abs(tderror) > clamp) {
      // huber loss to robustify
      if (tderror > clamp) tderror = clamp;
      if (tderror < -clamp) tderror = -clamp;
    }
    pred.dw[a0] = tderror;
    this.lastG.backward(); // compute gradients on net params

    // update net
    updateNet(this.net, this.alpha);
    return tderror;
  }
}

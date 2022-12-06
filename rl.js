var R = {}; // the Recurrent library

(function (global) {
  "use strict";
  // Mat holds a matrix

  var updateMat = function (m, alpha) {
    // updates in place
    for (var i = 0, n = m.n * m.d; i < n; i++) {
      if (m.dw[i] !== 0) {
        m.w[i] += -alpha * m.dw[i];
        m.dw[i] = 0;
      }
    }
  };

  var updateNet = function (net, alpha) {
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        updateMat(net[p], alpha);
      }
    }
  };

  var netToJSON = function (net) {
    var j = {};
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        j[p] = net[p].toJSON();
      }
    }
    return j;
  };
  var netFromJSON = function (j) {
    var net = {};
    for (var p in j) {
      if (j.hasOwnProperty(p)) {
        net[p] = new Mat(1, 1); // not proud of this
        net[p].fromJSON(j[p]);
      }
    }
    return net;
  };
  var netZeroGrads = function (net) {
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        var mat = net[p];
        gradFillConst(mat, 0);
      }
    }
  };
  var netFlattenGrads = function (net) {
    var n = 0;
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        var mat = net[p];
        n += mat.dw.length;
      }
    }
    var g = new Mat(n, 1);
    var ix = 0;
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        var mat = net[p];
        for (var i = 0, m = mat.dw.length; i < m; i++) {
          g.w[ix] = mat.dw[i];
          ix++;
        }
      }
    }
    return g;
  };

  var softmax = function (m) {
    var out = new Mat(m.n, m.d); // probability volume
    var maxval = -999999;
    for (var i = 0, n = m.w.length; i < n; i++) {
      if (m.w[i] > maxval) maxval = m.w[i];
    }

    var s = 0.0;
    for (var i = 0, n = m.w.length; i < n; i++) {
      out.w[i] = Math.exp(m.w[i] - maxval);
      s += out.w[i];
    }
    for (var i = 0, n = m.w.length; i < n; i++) {
      out.w[i] /= s;
    }

    // no backward pass here needed
    // since we will use the computed probabilities outside
    // to set gradients directly on m
    return out;
  };

  var Solver = function () {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  };
  Solver.prototype = {
    step: function (model, step_size, regc, clipval) {
      // perform parameter update
      var solver_stats = {};
      var num_clipped = 0;
      var num_tot = 0;
      for (var k in model) {
        if (model.hasOwnProperty(k)) {
          var m = model[k]; // mat ref
          if (!(k in this.step_cache)) {
            this.step_cache[k] = new Mat(m.n, m.d);
          }
          var s = this.step_cache[k];
          for (var i = 0, n = m.w.length; i < n; i++) {
            // rmsprop adaptive learning rate
            var mdwi = m.dw[i];
            s.w[i] =
              s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

            // gradient clip
            if (mdwi > clipval) {
              mdwi = clipval;
              num_clipped++;
            }
            if (mdwi < -clipval) {
              mdwi = -clipval;
              num_clipped++;
            }
            num_tot++;

            // update (and regularize)
            m.w[i] +=
              (-step_size * mdwi) / Math.sqrt(s.w[i] + this.smooth_eps) -
              regc * m.w[i];
            m.dw[i] = 0; // reset gradients for next iteration
          }
        }
      }
      solver_stats["ratio_clipped"] = (num_clipped * 1.0) / num_tot;
      return solver_stats;
    },
  };

  var initLSTM = function (input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    for (var d = 0; d < hidden_sizes.length; d++) {
      // loop over depths
      var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      var hidden_size = hidden_sizes[d];

      // gates parameters
      model["Wix" + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model["Wih" + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model["bi" + d] = new Mat(hidden_size, 1);
      model["Wfx" + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model["Wfh" + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model["bf" + d] = new Mat(hidden_size, 1);
      model["Wox" + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model["Woh" + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model["bo" + d] = new Mat(hidden_size, 1);
      // cell write params
      model["Wcx" + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model["Wch" + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model["bc" + d] = new Mat(hidden_size, 1);
    }
    // decoder params
    model["Whd"] = new RandMat(output_size, hidden_size, 0, 0.08);
    model["bd"] = new Mat(output_size, 1);
    return model;
  };

  var forwardLSTM = function (G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    if (prev == null || typeof prev.h === "undefined") {
      var hidden_prevs = [];
      var cell_prevs = [];
      for (var d = 0; d < hidden_sizes.length; d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d], 1));
        cell_prevs.push(new R.Mat(hidden_sizes[d], 1));
      }
    } else {
      var hidden_prevs = prev.h;
      var cell_prevs = prev.c;
    }

    var hidden = [];
    var cell = [];
    for (var d = 0; d < hidden_sizes.length; d++) {
      var input_vector = d === 0 ? x : hidden[d - 1];
      var hidden_prev = hidden_prevs[d];
      var cell_prev = cell_prevs[d];

      // input gate
      var h0 = G.mul(model["Wix" + d], input_vector);
      var h1 = G.mul(model["Wih" + d], hidden_prev);
      var input_gate = G.sigmoid(G.add(G.add(h0, h1), model["bi" + d]));

      // forget gate
      var h2 = G.mul(model["Wfx" + d], input_vector);
      var h3 = G.mul(model["Wfh" + d], hidden_prev);
      var forget_gate = G.sigmoid(G.add(G.add(h2, h3), model["bf" + d]));

      // output gate
      var h4 = G.mul(model["Wox" + d], input_vector);
      var h5 = G.mul(model["Woh" + d], hidden_prev);
      var output_gate = G.sigmoid(G.add(G.add(h4, h5), model["bo" + d]));

      // write operation on cells
      var h6 = G.mul(model["Wcx" + d], input_vector);
      var h7 = G.mul(model["Wch" + d], hidden_prev);
      var cell_write = G.tanh(G.add(G.add(h6, h7), model["bc" + d]));

      // compute new cell activation
      var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
      var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
      var cell_d = G.add(retain_cell, write_cell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

      hidden.push(hidden_d);
      cell.push(cell_d);
    }

    // one decoder to outputs at end
    var output = G.add(
      G.mul(model["Whd"], hidden[hidden.length - 1]),
      model["bd"]
    );

    // return cell memory, hidden representation and output
    return { h: hidden, c: cell, o: output };
  };

  // various utils
  global.maxi = maxi;
  global.samplei = samplei;
  global.randi = randi;
  global.randn = randn;
  global.softmax = softmax;
  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;
  global.forwardLSTM = forwardLSTM;
  global.initLSTM = initLSTM;
  // more utils
  global.updateMat = updateMat;
  global.updateNet = updateNet;
  global.copyMat = copyMat;
  global.copyNet = copyNet;
  global.netToJSON = netToJSON;
  global.netFromJSON = netFromJSON;
  global.netZeroGrads = netZeroGrads;
  global.netFlattenGrads = netFlattenGrads;
  // optimization
  global.Solver = Solver;
  global.Graph = Graph;
})(R);

// END OF RECURRENTJS

var RL = {};
(function (global) {
  "use strict";

  // syntactic sugar function for getting default parameter values
  var getopt = function (opt, field_name, default_value) {
    if (typeof opt === "undefined") {
      return default_value;
    }
    return typeof opt[field_name] !== "undefined"
      ? opt[field_name]
      : default_value;
  };

  var assert = R.assert;
  var randi = R.randi;

  // ------
  // AGENTS
  // ------

  var DQNAgent = function (env, opt) {
    this.gamma = getopt(opt, "gamma", 0.75); // future reward discount factor
    this.epsilon = getopt(opt, "epsilon", 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, "alpha", 0.01); // value function learning rate

    this.experience_add_every = getopt(opt, "experience_add_every", 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(opt, "experience_size", 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(
      opt,
      "learning_steps_per_iteration",
      10
    );
    this.tderror_clamp = getopt(opt, "tderror_clamp", 1.0);

    this.num_hidden_units = getopt(opt, "num_hidden_units", 100);

    this.env = env;
    this.reset();
  };
  DQNAgent.prototype = {
    reset: function () {
      this.nh = this.num_hidden_units; // number of hidden units
      this.ns = this.env.getNumStates();
      this.na = this.env.getMaxNumActions();

      // nets are hardcoded for now as key (str) -> Mat
      // not proud of this. better solution is to have a whole Net object
      // on top of Mats, but for now sticking with this
      this.net = {};
      this.net.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
      this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.net.W2 = new R.RandMat(this.na, this.nh, 0, 0.01);
      this.net.b2 = new R.Mat(this.na, 1, 0, 0.01);

      this.exp = []; // experience
      this.expi = 0; // where to insert

      this.t = 0;

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;

      this.tderror = 0; // for visualization only...
    },
    toJSON: function () {
      // save function
      var j = {};
      j.nh = this.nh;
      j.ns = this.ns;
      j.na = this.na;
      j.net = R.netToJSON(this.net);
      return j;
    },
    fromJSON: function (j) {
      // load function
      this.nh = j.nh;
      this.ns = j.ns;
      this.na = j.na;
      this.net = R.netFromJSON(j.net);
    },
    forwardQ: function (net, s, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      this.lastG = G; // back this up. Kind of hacky isn't it
      return a2mat;
    },
    act: function (slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.ns, 1);
      s.setFrom(slist);

      // epsilon greedy policy
      if (Math.random() < this.epsilon) {
        var a = randi(0, this.na);
      } else {
        // greedy wrt Q function
        var amat = this.forwardQ(this.net, s, false);
        var a = R.maxi(amat.w); // returns index of argmax action
      }

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    learn: function (r1) {
      // perform an update on Q function
      if (!(this.r0 == null) && this.alpha > 0) {
        // learn from this tuple to get a sense of how "surprising" it is to the agent
        var tderror = this.learnFromTuple(
          this.s0,
          this.a0,
          this.r0,
          this.s1,
          this.a1
        );
        this.tderror = tderror; // a measure of surprise

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
        for (var k = 0; k < this.learning_steps_per_iteration; k++) {
          var ri = randi(0, this.exp.length); // todo: priority sweeps?
          var e = this.exp[ri];
          this.learnFromTuple(e[0], e[1], e[2], e[3], e[4]);
        }
      }
      this.r0 = r1; // store for next update
    },
    learnFromTuple: function (s0, a0, r0, s1) {
      // want: Q(s,a) = r + gamma * max_a' Q(s',a')

      // compute the target Q value
      var tmat = this.forwardQ(this.net, s1, false);
      var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

      // now predict
      var pred = this.forwardQ(this.net, s0, true);

      var tderror = pred.w[a0] - qmax;
      var clamp = this.tderror_clamp;
      if (Math.abs(tderror) > clamp) {
        // huber loss to robustify
        if (tderror > clamp) tderror = clamp;
        if (tderror < -clamp) tderror = -clamp;
      }
      pred.dw[a0] = tderror;
      this.lastG.backward(); // compute gradients on net params

      // update net
      R.updateNet(this.net, this.alpha);
      return tderror;
    },
  };

  // exports
  global.DQNAgent = DQNAgent;
})(RL);

/*
简易版DQN的实现，基于环境奖励调整代理的连续行动策略。目前只支持一层隐藏层，足以应付简单AI游戏场景。
@author yuri2peter@qq.com
基于项目 https://github.com/karpathy/reinforcejs 修改简化
DQN示例 https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html

示例：
```
// 初始化
    var opt = {
    num_max_env_states: 8,
    num_max_env_actions: 4,
    num_hidden_units: 100,
    }
    var agent = new DQNAgent(opt); // opt是一个对象，支持若干参数 
    agent.adjust(opt); // 随时（除了关键参数）可以调整参数

    var envStates = []; // 当前环境状态，浮点数数组，其长度对应num_max_env_states，action为agent给出的操作序号。
    var action = 0; // 行动序号
    var reward = 0; // 奖励
    var smoothReward = 0; // 奖励平均值
    var tderror = 0; // 奖励预期偏差

// 迭代
    // ... 更新环境状态envStates
    action = agent.act(envStates); // 输入环境，agent给出action
    // ... 自行处理action对环境造成的影响并生成浮点数reward
    agent.learn(reward); // 让agent接受reward
    tderror = agent.tderror; // tderror体现了agent对于当前环境的奖励感到多么的“惊讶”（不符合预期）。
    smoothReward = smoothReward * 0.99 + reward; // 生成最近一段时间的平均奖励，越大表示学习效果越好。
    // ...更新环境、图表等可视化元素
// ...循环

// 随时导入导出训练结果
    var j = agent.toJSON();
    agent.fromJSON(j);
    
// 重置agent，将会丢失已有的训练成果
    agent.reset();
```

opt参数详解：
-----------
num_hidden_units: number of neurons in hidden layer 隐藏层神经元个数，默认100。【重要参数】，如果中途改动需要执行agent.reset()重置agent并使其生效。
num_max_env_states: 环境状态值的个数。【重要参数】，必选，如果中途改动需要执行agent.reset()重置agent并使其生效。
num_max_env_actions: agent可选操作的个数。【重要参数】，必选，如果中途改动需要执行agent.reset()重置agent并使其生效。
gamma: discount factor, [0, 1) 未来奖励折扣率，默认0.75。GAMMA值越高，表示我们希望agent更加关注未来，这比更加关注眼前更难，因此训练更加缓慢和困难。
epsilon: initial epsilon for epsilon-greedy policy, [0, 1) epsilon越高，agent的行为更富有随机性（冒险精神），默认0.1。
alpha: value function learning rate 学习率，默认0.01。需要开发者经验和试错来确定该参数。建议初期偏大，后期偏小。如果设置为0则完全放弃学习改进策略。
experience_size: size of experience replay memory 经验池大小，默认5000。如果经验回放集合尺寸太小了，必然要选择丢弃部分经验，如果选择丢弃的经验是很重要的，就会给训练带来不稳定。过大的尺寸会使训练计算量增大。
experience_add_every:  每隔N次learn()添加一次经验到经验池，可以设为1，默认25。默认不为1是为了节约资源并创造多样性。
learning_steps_per_iteration: 每次迭代的学习次数，默认10。该值越大效果越好，但是计算更慢。
tderror_clamp: for robustness ；该值规定了tderror绝对值的上限，默认1.0。tderror体现了agent对于当前环境的奖励感到多么的“惊讶”（不符合预期）。
-----------
*/


(function (g2) {
    "use strict";

    var R = {}; // the Recurrent library

    (function (g1) {
        "use strict";

        // Utility fun
        function assert(condition, message) {
            // from http://stackoverflow.com/questions/15313418/javascript-assert
            if (!condition) {
                message = message || "Assertion failed";
                if (typeof Error !== "undefined") {
                    throw new Error(message);
                }
                throw message; // Fallback
            }
        }

        // Random numbers utils
        var return_v = false;
        var v_val = 0.0;
        var gaussRandom = function () {
            if (return_v) {
                return_v = false;
                return v_val;
            }
            var u = 2 * Math.random() - 1;
            var v = 2 * Math.random() - 1;
            var r = u * u + v * v;
            if (r == 0 || r > 1) return gaussRandom();
            var c = Math.sqrt(-2 * Math.log(r) / r);
            v_val = v * c; // cache this
            return_v = true;
            return u * c;
        }
        var randf = function (a, b) { return Math.random() * (b - a) + a; }
        var randi = function (a, b) { return Math.floor(Math.random() * (b - a) + a); }
        var randn = function (mu, std) { return mu + gaussRandom() * std; }

        // helper function returns array of zeros of length n
        // and uses typed arrays if available
        var zeros = function (n) {
            if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
            if (typeof ArrayBuffer === 'undefined') {
                // lacking browser support
                var arr = new Array(n);
                for (var i = 0; i < n; i++) { arr[i] = 0; }
                return arr;
            } else {
                return new Float64Array(n);
            }
        }

        // Mat holds a matrix
        var Mat = function (n, d) {
            // n is number of rows d is number of columns
            this.n = n;
            this.d = d;
            this.w = zeros(n * d);
            this.dw = zeros(n * d);
        }
        Mat.prototype = {
            get: function (row, col) {
                // slow but careful accessor function
                // we want row-major order
                var ix = (this.d * row) + col;
                assert(ix >= 0 && ix < this.w.length);
                return this.w[ix];
            },
            set: function (row, col, v) {
                // slow but careful accessor function
                var ix = (this.d * row) + col;
                assert(ix >= 0 && ix < this.w.length);
                this.w[ix] = v;
            },
            setFrom: function (arr) {
                for (var i = 0, n = arr.length; i < n; i++) {
                    this.w[i] = arr[i];
                }
            },
            setColumn: function (m, i) {
                for (var q = 0, n = m.w.length; q < n; q++) {
                    this.w[(this.d * q) + i] = m.w[q];
                }
            },
            toJSON: function () {
                var json = {};
                json['n'] = this.n;
                json['d'] = this.d;
                json['w'] = this.w;
                return json;
            },
            fromJSON: function (json) {
                this.n = json.n;
                this.d = json.d;
                this.w = zeros(this.n * this.d);
                this.dw = zeros(this.n * this.d);
                for (var i = 0, n = this.n * this.d; i < n; i++) {
                    this.w[i] = json.w[i]; // copy over weights
                }
            }
        }

        var copyMat = function (b) {
            var a = new Mat(b.n, b.d);
            a.setFrom(b.w);
            return a;
        }

        var copyNet = function (net) {
            // nets are (k,v) pairs with k = string key, v = Mat()
            var new_net = {};
            for (var p in net) {
                if (net.hasOwnProperty(p)) {
                    new_net[p] = copyMat(net[p]);
                }
            }
            return new_net;
        }

        var updateMat = function (m, alpha) {
            // updates in place
            for (var i = 0, n = m.n * m.d; i < n; i++) {
                if (m.dw[i] !== 0) {
                    m.w[i] += - alpha * m.dw[i];
                    m.dw[i] = 0;
                }
            }
        }

        var updateNet = function (net, alpha) {
            for (var p in net) {
                if (net.hasOwnProperty(p)) {
                    updateMat(net[p], alpha);
                }
            }
        }

        var netToJSON = function (net) {
            var j = {};
            for (var p in net) {
                if (net.hasOwnProperty(p)) {
                    j[p] = net[p].toJSON();
                }
            }
            return j;
        }
        var netFromJSON = function (j) {
            var net = {};
            for (var p in j) {
                if (j.hasOwnProperty(p)) {
                    net[p] = new Mat(1, 1); // not proud of this
                    net[p].fromJSON(j[p]);
                }
            }
            return net;
        }
        var netZeroGrads = function (net) {
            for (var p in net) {
                if (net.hasOwnProperty(p)) {
                    var mat = net[p];
                    gradFillConst(mat, 0);
                }
            }
        }
        var netFlattenGrads = function (net) {
            var n = 0;
            for (var p in net) { if (net.hasOwnProperty(p)) { var mat = net[p]; n += mat.dw.length; } }
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
        }

        // return Mat but filled with random numbers from gaussian
        var RandMat = function (n, d, mu, std) {
            var m = new Mat(n, d);
            fillRandn(m, mu, std);
            //fillRand(m,-std,std); // kind of :P
            return m;
        }

        // Mat utils
        // fill matrix with random gaussian numbers
        var fillRandn = function (m, mu, std) { for (var i = 0, n = m.w.length; i < n; i++) { m.w[i] = randn(mu, std); } }
        var fillRand = function (m, lo, hi) { for (var i = 0, n = m.w.length; i < n; i++) { m.w[i] = randf(lo, hi); } }
        var gradFillConst = function (m, c) { for (var i = 0, n = m.dw.length; i < n; i++) { m.dw[i] = c } }

        // Transformer definitions
        var Graph = function (needs_backprop) {
            if (typeof needs_backprop === 'undefined') { needs_backprop = true; }
            this.needs_backprop = needs_backprop;

            // this will store a list of functions that perform backprop,
            // in their forward pass order. So in backprop we will go
            // backwards and evoke each one
            this.backprop = [];
        }
        Graph.prototype = {
            backward: function () {
                for (var i = this.backprop.length - 1; i >= 0; i--) {
                    this.backprop[i](); // tick!
                }
            },
            rowPluck: function (m, ix) {
                // pluck a row of m with index ix and return it as col vector
                assert(ix >= 0 && ix < m.n);
                var d = m.d;
                var out = new Mat(d, 1);
                for (var i = 0, n = d; i < n; i++) { out.w[i] = m.w[d * ix + i]; } // copy over the data

                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0, n = d; i < n; i++) { m.dw[d * ix + i] += out.dw[i]; }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            tanh: function (m) {
                // tanh nonlinearity
                var out = new Mat(m.n, m.d);
                var n = m.w.length;
                for (var i = 0; i < n; i++) {
                    out.w[i] = Math.tanh(m.w[i]);
                }

                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0; i < n; i++) {
                            // grad for z = tanh(x) is (1 - z^2)
                            var mwi = out.w[i];
                            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            sigmoid: function (m) {
                // sigmoid nonlinearity
                var out = new Mat(m.n, m.d);
                var n = m.w.length;
                for (var i = 0; i < n; i++) {
                    out.w[i] = sig(m.w[i]);
                }

                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0; i < n; i++) {
                            // grad for z = tanh(x) is (1 - z^2)
                            var mwi = out.w[i];
                            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            relu: function (m) {
                var out = new Mat(m.n, m.d);
                var n = m.w.length;
                for (var i = 0; i < n; i++) {
                    out.w[i] = Math.max(0, m.w[i]); // relu
                }
                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0; i < n; i++) {
                            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            mul: function (m1, m2) {
                // multiply matrices m1 * m2
                assert(m1.d === m2.n, 'matmul dimensions misaligned');

                var n = m1.n;
                var d = m2.d;
                var out = new Mat(n, d);
                for (var i = 0; i < m1.n; i++) { // loop over rows of m1
                    for (var j = 0; j < m2.d; j++) { // loop over cols of m2
                        var dot = 0.0;
                        for (var k = 0; k < m1.d; k++) { // dot product loop
                            dot += m1.w[m1.d * i + k] * m2.w[m2.d * k + j];
                        }
                        out.w[d * i + j] = dot;
                    }
                }

                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0; i < m1.n; i++) { // loop over rows of m1
                            for (var j = 0; j < m2.d; j++) { // loop over cols of m2
                                for (var k = 0; k < m1.d; k++) { // dot product loop
                                    var b = out.dw[d * i + j];
                                    m1.dw[m1.d * i + k] += m2.w[m2.d * k + j] * b;
                                    m2.dw[m2.d * k + j] += m1.w[m1.d * i + k] * b;
                                }
                            }
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            add: function (m1, m2) {
                assert(m1.w.length === m2.w.length);

                var out = new Mat(m1.n, m1.d);
                for (var i = 0, n = m1.w.length; i < n; i++) {
                    out.w[i] = m1.w[i] + m2.w[i];
                }
                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0, n = m1.w.length; i < n; i++) {
                            m1.dw[i] += out.dw[i];
                            m2.dw[i] += out.dw[i];
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            dot: function (m1, m2) {
                // m1 m2 are both column vectors
                assert(m1.w.length === m2.w.length);
                var out = new Mat(1, 1);
                var dot = 0.0;
                for (var i = 0, n = m1.w.length; i < n; i++) {
                    dot += m1.w[i] * m2.w[i];
                }
                out.w[0] = dot;
                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0, n = m1.w.length; i < n; i++) {
                            m1.dw[i] += m2.w[i] * out.dw[0];
                            m2.dw[i] += m1.w[i] * out.dw[0];
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
            eltmul: function (m1, m2) {
                assert(m1.w.length === m2.w.length);

                var out = new Mat(m1.n, m1.d);
                for (var i = 0, n = m1.w.length; i < n; i++) {
                    out.w[i] = m1.w[i] * m2.w[i];
                }
                if (this.needs_backprop) {
                    var backward = function () {
                        for (var i = 0, n = m1.w.length; i < n; i++) {
                            m1.dw[i] += m2.w[i] * out.dw[i];
                            m2.dw[i] += m1.w[i] * out.dw[i];
                        }
                    }
                    this.backprop.push(backward);
                }
                return out;
            },
        }

        var softmax = function (m) {
            var out = new Mat(m.n, m.d); // probability volume
            var maxval = -999999;
            for (var i = 0, n = m.w.length; i < n; i++) { if (m.w[i] > maxval) maxval = m.w[i]; }

            var s = 0.0;
            for (var i = 0, n = m.w.length; i < n; i++) {
                out.w[i] = Math.exp(m.w[i] - maxval);
                s += out.w[i];
            }
            for (var i = 0, n = m.w.length; i < n; i++) { out.w[i] /= s; }

            // no backward pass here needed
            // since we will use the computed probabilities outside
            // to set gradients directly on m
            return out;
        }

        var Solver = function () {
            this.decay_rate = 0.999;
            this.smooth_eps = 1e-8;
            this.step_cache = {};
        }
        Solver.prototype = {
            step: function (model, step_size, regc, clipval) {
                // perform parameter update
                var solver_stats = {};
                var num_clipped = 0;
                var num_tot = 0;
                for (var k in model) {
                    if (model.hasOwnProperty(k)) {
                        var m = model[k]; // mat ref
                        if (!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
                        var s = this.step_cache[k];
                        for (var i = 0, n = m.w.length; i < n; i++) {

                            // rmsprop adaptive learning rate
                            var mdwi = m.dw[i];
                            s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

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
                            m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
                            m.dw[i] = 0; // reset gradients for next iteration
                        }
                    }
                }
                solver_stats['ratio_clipped'] = num_clipped * 1.0 / num_tot;
                return solver_stats;
            }
        }

        var initLSTM = function (input_size, hidden_sizes, output_size) {
            // hidden size should be a list

            var model = {};
            for (var d = 0; d < hidden_sizes.length; d++) { // loop over depths
                var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
                var hidden_size = hidden_sizes[d];

                // gates parameters
                model['Wix' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
                model['Wih' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
                model['bi' + d] = new Mat(hidden_size, 1);
                model['Wfx' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
                model['Wfh' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
                model['bf' + d] = new Mat(hidden_size, 1);
                model['Wox' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
                model['Woh' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
                model['bo' + d] = new Mat(hidden_size, 1);
                // cell write params
                model['Wcx' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
                model['Wch' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
                model['bc' + d] = new Mat(hidden_size, 1);
            }
            // decoder params
            model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
            model['bd'] = new Mat(output_size, 1);
            return model;
        }

        var forwardLSTM = function (G, model, hidden_sizes, x, prev) {
            // forward prop for a single tick of LSTM
            // G is graph to append ops to
            // model contains LSTM parameters
            // x is 1D column vector with observation
            // prev is a struct containing hidden and cell
            // from previous iteration

            if (prev == null || typeof prev.h === 'undefined') {
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
                var h0 = G.mul(model['Wix' + d], input_vector);
                var h1 = G.mul(model['Wih' + d], hidden_prev);
                var input_gate = G.sigmoid(G.add(G.add(h0, h1), model['bi' + d]));

                // forget gate
                var h2 = G.mul(model['Wfx' + d], input_vector);
                var h3 = G.mul(model['Wfh' + d], hidden_prev);
                var forget_gate = G.sigmoid(G.add(G.add(h2, h3), model['bf' + d]));

                // output gate
                var h4 = G.mul(model['Wox' + d], input_vector);
                var h5 = G.mul(model['Woh' + d], hidden_prev);
                var output_gate = G.sigmoid(G.add(G.add(h4, h5), model['bo' + d]));

                // write operation on cells
                var h6 = G.mul(model['Wcx' + d], input_vector);
                var h7 = G.mul(model['Wch' + d], hidden_prev);
                var cell_write = G.tanh(G.add(G.add(h6, h7), model['bc' + d]));

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
            var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]), model['bd']);

            // return cell memory, hidden representation and output
            return { 'h': hidden, 'c': cell, 'o': output };
        }

        var sig = function (x) {
            // helper function for computing sigmoid
            return 1.0 / (1 + Math.exp(-x));
        }

        var maxi = function (w) {
            // argmax of array w
            var maxv = w[0];
            var maxix = 0;
            for (var i = 1, n = w.length; i < n; i++) {
                var v = w[i];
                if (v > maxv) {
                    maxix = i;
                    maxv = v;
                }
            }
            return maxix;
        }

        var samplei = function (w) {
            // sample argmax from w, assuming w are 
            // probabilities that sum to one
            var r = randf(0, 1);
            var x = 0.0;
            var i = 0;
            while (true) {
                x += w[i];
                if (x > r) { return i; }
                i++;
            }
            return w.length - 1; // pretty sure we should never get here?
        }

        // various utils
        g1.assert = assert;
        g1.zeros = zeros;
        g1.maxi = maxi;
        g1.samplei = samplei;
        g1.randi = randi;
        g1.randn = randn;
        g1.softmax = softmax;
        // classes
        g1.Mat = Mat;
        g1.RandMat = RandMat;
        g1.forwardLSTM = forwardLSTM;
        g1.initLSTM = initLSTM;
        // more utils
        g1.updateMat = updateMat;
        g1.updateNet = updateNet;
        g1.copyMat = copyMat;
        g1.copyNet = copyNet;
        g1.netToJSON = netToJSON;
        g1.netFromJSON = netFromJSON;
        g1.netZeroGrads = netZeroGrads;
        g1.netFlattenGrads = netFlattenGrads;
        // optimization
        g1.Solver = Solver;
        g1.Graph = Graph;
    })(R);

    // syntactic sugar function for getting default parameter values
    var getopt = function (opt, field_name, default_value) {
        if (typeof opt === 'undefined') { return default_value; }
        return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
    }

    var randi = R.randi;

    // ------
    // AGENTS
    // ------

    var DQNAgent = function (opt) {
        this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
        this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
        this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

        this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
        this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
        this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
        this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);

        this.num_hidden_units = getopt(opt, 'num_hidden_units', 100);
        this.num_max_env_states = opt.num_max_env_states;
        this.num_max_env_actions = opt.num_max_env_actions;

        this.reset();
    }
    DQNAgent.prototype = {
        // 调整参数
        // 可接受的参数：gamma, epsilon, alpha, experience_add_every, experience_size, learning_steps_per_iteration, tderror_clamp, num_hidden_units, num_max_env_states, num_max_env_actions
        // 注意，num_max_env_states, num_max_env_actions, num_hidden_units的改变必须要在reset()后生效
        adjust: function (opt) {
            var pList = [
                'gamma', 'epsilon', 'alpha', 'experience_add_every', 'experience_size', 'learning_steps_per_iteration',
                'tderror_clamp', 'num_hidden_units', 'num_max_env_states', 'num_max_env_actions'
            ]
            Object.getOwnPropertyNames(opt).forEach(function (p) {
                if (pList.indexOf(p) >= 0) {
                    this[p] = opt[p];
                }
            })
        },
        reset: function () {
            this.nh = this.num_hidden_units; // number of hidden units
            this.ns = this.num_max_env_states;
            this.na = this.num_max_env_actions;

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
                var tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
                this.tderror = tderror; // a measure of surprise

                // decide if we should keep this experience in the replay
                if (this.t % this.experience_add_every === 0) {
                    this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
                    this.expi += 1;
                    if (this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
                }
                this.t += 1;

                // sample some additional experience from replay memory and learn from it
                for (var k = 0; k < this.learning_steps_per_iteration; k++) {
                    var ri = randi(0, this.exp.length); // todo: priority sweeps?
                    var e = this.exp[ri];
                    this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
                }
            }
            this.r0 = r1; // store for next update
        },
        learnFromTuple: function (s0, a0, r0, s1, a1) {
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')

            // compute the target Q value
            var tmat = this.forwardQ(this.net, s1, false);
            var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

            // now predict
            var pred = this.forwardQ(this.net, s0, true);

            var tderror = pred.w[a0] - qmax;
            var clamp = this.tderror_clamp;
            if (Math.abs(tderror) > clamp) {  // huber loss to robustify
                if (tderror > clamp) tderror = clamp;
                if (tderror < -clamp) tderror = -clamp;
            }
            pred.dw[a0] = tderror;
            this.lastG.backward(); // compute gradients on net params

            // update net
            R.updateNet(this.net, this.alpha);
            return tderror;
        }
    }


    // exports
    g2.DQNAgent = DQNAgent;
})(globalThis || window || global);


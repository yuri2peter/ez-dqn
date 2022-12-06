# ez-dqn

"ez-dqn" 是 DQN 强化学习的简单实现，提供了一个相对精简的 API，在小型游戏场景下有较好的表现。
源码基于[karpathy/reinforcejs](https://github.com/karpathy/reinforcejs/blob/master/lib/rl.js)改编。

## 安装

`npm i @yuri2/ez-dqn`

## 使用

### Quick Start

```ts
import { srctest as sleep } from "../src/index";

async function main() {
  console.log(1);
  await sleep(1000);
  console.log(2);
}

main();
```

### opt 参数详解

| 参数                         | 意义                                                                                                                                                                                         |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| num_hidden_units             | number of neurons in hidden layer 隐藏层神经元个数，默认 100                                                                                                                                 |
| num_max_env_states           | 环境状态值的个数                                                                                                                                                                             |
| num_max_env_actions          | agent 可选操作的个数                                                                                                                                                                         |
| gamma                        | discount factor, [0, 1) 未来奖励折扣率，默认 0.75。GAMMA 值越高，表示我们希望 agent 更加关注未来，这比更加关注眼前更难，因此训练更加缓慢和困难                                               |
| epsilon                      | initial epsilon for epsilon-greedy policy, [0, 1) epsilon 越高，agent 的行为更富有随机性（冒险精神），默认 0.1                                                                               |
| alpha                        | value function learning rate 学习率，默认 0.01。需要开发者经验和试错来确定该参数。建议初期偏大，后期偏小。如果设置为 0 则完全放弃学习改进策略                                                |
| experience_size              | size of experience replay memory 经验池大小，默认 5000。如果经验回放集合尺寸太小了，必然要选择丢弃部分经验，如果选择丢弃的经验是很重要的，就会给训练带来不稳定。过大的尺寸会使训练计算量增大 |
| experience_add_every         | 每隔 N 次 learn()添加一次经验到经验池，可以设为 1，默认 25。默认不为 1 是为了节约资源并创造多样性                                                                                            |
| learning_steps_per_iteration | 每次迭代的学习次数，默认 10。该值越大效果越好，但是计算更慢                                                                                                                                  |
| tderror_clamp                | 该值规定了 tderror 绝对值的上限，默认 1.0。tderror 体现了 agent 对于当前环境的奖励感到多么的“惊讶”（不符合预期）                                                                             |

## 开发

### 源码编写

- `src` 目录下编写源码
- `test` 目录下编写测试
- `npm run test` 执行测试文件 `test/index.ts`

### 打包编译

```
按需求，修改rollup.config.js文件
npm run build 生成index.ts文件和.d.ts声明文件
```

### 发布前测试

1. 全局测试：把包链接到全局环境
   ` npm link`

2. 本地项目测试：把包链接到项目本地环境
   `cd 本地项目根目录`
   `npm link 包名`

3. 取消本地项目测试：把包从本地环境取消
   `cd 本地项目根目录`
   `npm unlink 包名`

4. 取消全局测试：把包从全局环境中取消
   `npm unlink`

### npm 发布

第一次发布：

- 修改版本号
- 提交 github
  `npm publish --access public`

更新版本：
`npm run release`

# ez-dqn

"ez-dqn" 是 DQN 强化学习的简单实现，提供了一个相对精简的 API，在小型游戏场景下有较好的表现。
源码基于[karpathy/reinforcejs](https://github.com/karpathy/reinforcejs/blob/master/lib/rl.js)改编。

## 安装

`npm i @yuri2/ez-dqn`

## 使用

```ts
import { srctest as sleep } from "../src/index";

async function main() {
  console.log(1);
  await sleep(1000);
  console.log(2);
}

main();
```

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

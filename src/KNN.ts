import {
  Tensor,
  tensor,
  randomNormal,
  sequential,
  relu,
  tanh,
  scalar,
  div,
  sqrt,
  zeros,
  sum,
  exp,
  square,
  range,
  argMin,
  sub,
  softmax,
  sigmoid,
  tensor1d,
  bincount,
} from '@tensorflow/tfjs'
import { most } from './m'
export class KNearestNeighbor {
  x!: Tensor
  y!: Tensor
  constructor(public reg: 'L1' | 'L2') {}
  train(x: Tensor, y: Tensor) {
    this.x = x
    this.y = y
  }

  predict(testX: Tensor, k: number = 1) {
    const nt = testX.shape[0]
    let Ypred = zeros([nt]).arraySync() as number[]
    const testXArray = testX.arraySync() as number[][]
    const trainXArray = this.x.arraySync() as number[][]
    const y = this.y.arraySync() as number[]
    // 遍历测试集
    for (const i in range(0, nt).arraySync()) {
      let ds: number[] = []
      for (const trainI of trainXArray) {
        let d = tensor(trainI).sub(testXArray[i])

        if (this.reg == 'L1') {
          d = d.abs().sum()
        } else {
          d = d.square().sum().sqrt()
        }
        ds.push(d.arraySync() as number)
      }

      let min_index: number

      if (k == 1) {
        // 如果k=1取最近的一个
        min_index = tensor1d(ds).argMin().arraySync() as number
        Ypred[i] = y[min_index]
      } else {
        // 否则取最近的k个里标签出现最多的，有同样多的就取距离最近的
        console.log('dist', ds)

        const dist_k = [...ds]
          .sort((a, b) => a - b)
          .slice(0, k)
          .sort((a, b) => b - a)

        console.log('dist_k', dist_k)
        console.log('y label', y.flat())

        const predLabels = dist_k.map((v) => y.flat()[ds.indexOf(v)]).flat()
        console.log('predLabels', predLabels)

        Ypred[i] = most(predLabels) as number
      }
    }
    return Ypred
  }
}

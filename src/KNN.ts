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
      } else {
        // 否则取最近的k个里标签出现最多的，有同样多的就取距离最近的
        const yl = [...ds]
          .sort((a, b) => a - b)
          .slice(0, k)
          .sort((a, b) => b - a)

        console.log('yl', yl)
        const tya = this.y.arraySync() as number[]
        console.log('tya', tya.flat())
        console.log('ds', ds)

        const ys = yl.map((v) => tya.flat()[ds.indexOf(v)]).flat()
        console.log('ys', ys)

        min_index = most(ys) as number
        console.log('min_index', min_index)
      }
      Ypred[i] = (this.y.arraySync() as number[])[min_index]
    }
    return Ypred
  }
}

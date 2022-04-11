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
  sum,
  exp,
  sub,
  softmax,
  sigmoid,
  Shape,
  range,
  transpose,
  zeros,
  add,
  TensorLike,
  TensorContainer,
  tidy,
  data,
} from '@tensorflow/tfjs'
import { zip } from './m'

import { randomReLUWeight } from './weight'

type Data = {
  xs: TensorContainer[]
  ys: TensorContainer[]
}
export class NN {
  public num_layers = 0
  public weights!: Tensor[]
  public biases!: Tensor[]
  constructor(public shape: number[]) {
    this.num_layers = shape.length

    // 获取bias和weights的shape分量
    const shape_w = shape.slice(0, -1)
    console.log('shape_w', shape_w)
    const shape_b = shape.slice(1)
    console.log('shape_b', shape_b)

    // 初始化bias并打印
    this.biases = shape_b.map((v) => randomNormal([v, 1]))
    console.log('biases:')
    this.biases.map((v, i) => {
      console.log('b' + i)

      v.print()
    })

    // 初始化weights并打印

    this.weights = zip(shape_w, shape_b).map(([x, y], i) => {
      return randomNormal([y, x])
    })
    console.log('weights:')
    this.weights.map((v, i) => {
      console.log('w' + i)
      v.print()
    })
  }
  forward(inputs: Tensor) {
    // // w*x
    // const wx = this.weights.dot(inputs)
    // // y = w*x+b
    // const y = wx.add(this.biases)
    // // active
    // const out = relu(y)
    // out.print()
    // return out
  }
  sgd(trainData: Data[], epochs: number, miniBatchSize: number, eta: number) {
    for (let i = 0; i < epochs; i++) {
      // 随机batch
      for (let j = 0; j < trainData.length; j += miniBatchSize) {
        console.log(trainData.slice(j, miniBatchSize))

        this.updateMiniBatch(trainData.slice(j, miniBatchSize), eta)
      }
      console.log(`Epoch ${i} complete`)
    }
  }

  updateMiniBatch(miniBatch: Data[], eta: number) {
    let nb = this.biases.map((b) => {
      return zeros(b.shape)
    })
    console.log('nabla_b', nb)

    let nw = this.weights.map((w) => {
      return zeros(w.shape)
    })
    console.log('nabla_w', nw)
    console.log('miniBatch', miniBatch)

    miniBatch.map((v) => {
      const [dnb, dnw] = this.backprop(v)
      // nb = zip(nb, dnb).map(([a, b]) => add(a, b))
      // nw = zip(nw, dnw).map(([a, b]) => add(a, b))
      // this.weights = zip(this.weights, nw).map(([w, nw]) => w - (eta / miniBatch.length) * nw) as unknown as Tensor[]
      // this.biases = zip(this.biases, nw).map(([b, nb]) => b - (eta / miniBatch.length) * nb) as unknown as Tensor[]
    })
  }

  backprop(miniBatch: Data) {
    const { xs: x, ys: y } = miniBatch
    const nb = this.biases.map((b) => zeros(b.shape))
    const nw = this.weights.map((w) => zeros(w.shape))
    // feedforward

    tensor(x as number[]).print()
    let tx: any = tensor(x as number[])
    let as = [x]
    const zs = []
    zip(this.biases, this.weights).map(([b, w]: [Tensor, Tensor], i) => {
      console.log(i + ':')

      console.log('w')
      w.print()
      console.log('a')
      tx.print()
      const wa = w.dot(tx)
      console.log('wa')
      wa.print()
      const z = wa.add(b)
      console.log('z')
      z.print()

      zs.push(z)
      tx = z.sigmoid()
      console.log('sigmoid(x):')
      tx.print()
      as.push(tx)
    })
    //backward
    return [[0], [0]]
  }
  evaluate(testData: Tensor) {}
}

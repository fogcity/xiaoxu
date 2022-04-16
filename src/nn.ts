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
  dot,
} from '@tensorflow/tfjs'
import { zip } from './m'
import { log } from './utils'

import { randomReLUWeight } from './weight'

type Data = {
  xs: TensorLike[]
  ys: TensorLike[]
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

    this.weights = zip(shape_w, shape_b).map(([x, y]) => {
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
    console.log('miniBatch', miniBatch)

    const { xs: x, ys: y } = miniBatch
    const nb = this.biases.map((b) => zeros(b.shape))
    const nw = this.weights.map((w) => zeros(w.shape))
    // 前馈开始
    let tx: any = tensor(x as number[])
    let ty: any = tensor(y as number[])
    let as = [tx]
    let zs: Tensor[] = []
    zip(this.biases, this.weights).map(([b, w]: [Tensor, Tensor], i) => {
      console.log(i + ':')

      console.log('w')
      w.print()
      console.log('x')
      tx.print()
      const wx = w.dot(tx)
      console.log('w*x')
      wx.print()
      console.log('b')
      b.print()
      const z = wx.add(b)
      console.log('w*x+b')
      z.print()

      zs.push(z)
      tx = z.sigmoid()
      console.log('sigmoid(w*x+b):')
      tx.print()
      as.push(tx)
    })

    // 查看前馈结果
    log('forward result:')
    log('zs')
    zs.map((v) => v.print())
    log('as')
    as.map((v) => v.print())

    // 反向传播开始
    this.cost_derivative(as[as.length - 1], ty).print()

    this.sigmoid_derivative(zs[zs.length]).print()
    let delta = this.cost_derivative(as[as.length - 1], ty).mul(this.sigmoid_derivative(zs[zs.length]))
    log('delta')
    delta.print()
    nb[nb.length - 1] = delta
    nw[nw.length - 1] = delta.dot(as[nw.length - 2].transpose())

    range(2, this.num_layers)
      .arraySync()
      .map((l) => {
        const z = zs[nb.length - l]
        const sp = this.sigmoid_derivative(z)
        delta = dot(this.weights[-l + 1].transpose(), delta).mul(sp)
        nb[nb.length - 1] = delta
        nb[nb.length - 1] = dot(delta, as[-l - 1].transpose())
      })

    return [nb, nw]
  }
  cost_derivative(output_activations: Tensor, y: Tensor) {
    // C = 12 ∑j (yj − aj )2  =>  ∂C/∂aLj = (aj − yj )，

    return output_activations.sub(y)
  }
  sigmoid_derivative(z: Tensor) {
    return z.sigmoid().mul(scalar(1).sub(z.sigmoid()))
  }

  evaluate(testData: Tensor) {}
}

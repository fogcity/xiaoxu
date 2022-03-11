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
} from '@tensorflow/tfjs'
export class Neuron {
  public weights!: Tensor
  public bias: number = 0
  constructor() {}
  forward(inputs: Tensor) {
    console.log('input.shape', inputs.shape)
    const r = sqrt(div(2.0, inputs.shape))
    const w = randomNormal(inputs.shape)

    !this.weights && (this.weights = w.mul(r))
    console.log('#w', this.weights.toString())
    console.log('#input', inputs.toString())

    // w*x
    const wx = this.weights.dot(inputs)
    // y = w*x+b
    const y = wx.add(this.bias)
    console.log('y', y.toString())

    // active
    const out = relu(y)
    console.log('relu', out.toString())
    return out
  }
}

const n = new Neuron()
n.forward(tensor([2, 2, 3]))

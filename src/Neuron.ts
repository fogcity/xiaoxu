import {
  Tensor,
  tensor,
  randomNormal,
  sequential,
  relu,
  tanh,
  scalar,
  div,
  sum,
  exp,
  sub,
  softmax,
  sigmoid,
} from '@tensorflow/tfjs'
export class Neuron {
  public weights!: Tensor
  public bias!: number
  constructor() {
    this.bias = randomNormal([]).arraySync() as number
  }
  forward(inputs: Tensor) {
    !this.weights && (this.weights = randomNormal(inputs.shape))
    console.log('W', this.weights.toString())
    console.log('b', this.bias)
    console.log('input', inputs.toString())

    const cell_body_sum = this.weights.dot(inputs).add(this.bias)
    console.log('cell_body_sum', cell_body_sum.toString())

    console.log('cell_body_sum.mul(-1)', cell_body_sum.mul(-1).toString())

    // sigmoid = scalar(1.0).div(exp(-cell_body_sum).add(1.0))

    const a = exp(cell_body_sum.mul(-1))
    console.log('a', a.toString())
    const d = sub(1, a)
    console.log('d', d.toString())

    console.log('sss', div(1.0, d).toString())
    let firing_rate = sigmoid(cell_body_sum)
    console.log('sigmoid', firing_rate.toString())
    firing_rate = relu(cell_body_sum)
    console.log('relu', firing_rate.toString())
    firing_rate = tanh(cell_body_sum)
    console.log('tanh', firing_rate.toString())

    return firing_rate
  }
}

const n = new Neuron()
n.forward(tensor([2, 2]))

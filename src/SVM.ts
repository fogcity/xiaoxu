import { Unit } from './Gate'
import { Circuit, createCircuit } from './Circuit'

// SVM:
export interface SVM {
  learnFrom(x: Unit, y: Unit, label: number): void
  a: Unit
  b: Unit
  c: Unit
  circuit: Circuit
  forward(x: Unit, y: Unit): Unit
  backward(label: number): void
}
export type SVMCreator = { new (): SVM }

const SVM = function (this: SVM) {
  // SVM class -> f(x,y)=ax+by+c .
  // random initial parameter values

  // 随机化参数
  this.a = new Unit(-2.34, 0.0)
  this.b = new Unit(1.22, 0.0)
  this.c = new Unit(-0.53, 0.0)

  this.circuit = createCircuit()
} as unknown as SVMCreator

SVM.prototype = {
  forward: function (x: number, y: number) {
    // assume x and y are Units
    this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c)
    return this.unit_out
  },
  backward: function (label: number) {
    // 重置所有的梯度
    this.a.grad = 0.0
    this.b.grad = 0.0
    this.c.grad = 0.0

    // 计算对应的拉力
    let pull = 0.0
    if (label === 1 && this.unit_out.value < 1) {
      pull = 1 // the score was too low: pull up
    }
    if (label === -1 && this.unit_out.value > -1) {
      pull = -1 // the score was too high for a positive example, pull down
    }

    this.circuit.backward(pull) // writes gradient into x,y,a,b,c

    // add regularization pull for parameters: towards zero and proportional to value
    this.a.grad += -this.a.value
    this.b.grad += -this.b.value
  },
  learnFrom: function (x: number, y: number, label: number) {
    this.forward(x, y) // forward pass (set .value in all Units)
    this.backward(label) // backward pass (set .grad in all Units)
    this.parameterUpdate() // parameters respond to tug
  },
  parameterUpdate: function () {
    const step_size = 0.0001
    this.a.value += step_size * this.a.grad
    this.b.value += step_size * this.b.grad
    this.c.value += step_size * this.c.grad
  },
}

const data: number[][] = [],
  labels: number[] = [-1, -1, 1, -1]
data.push([1.2, 0.7])
data.push([1.6, 0.1])
data.push([-0.3, -0.5])
data.push([-0.7, -0.1])

console.table(data)

// 初始化svm
const svm = new SVM()

// a function that computes the classification accuracy
const evalTrainingAccuracy = () => {
  let num_correct = 0
  for (let i = 0; i < data.length; i++) {
    const x = new Unit(data[i][0], 0.0)
    const y = new Unit(data[i][1], 0.0)
    const true_label = labels[i]

    // see if the prediction matches the provided label
    const v = svm.forward(x, y).value

    const predicted_label = v > 0 ? 1 : -1

    if (predicted_label === true_label) {
      num_correct++
    }
  }

  return num_correct / data.length
}

// the learning loop
for (let iter = 0; iter < 100000; iter++) {
  // pick a random data point
  const i = Math.floor(Math.random() * data.length)

  const x = new Unit(data[i][0], 0.0)
  const y = new Unit(data[i][1], 0.0)

  const label = labels[i]

  svm.learnFrom(x, y, label)

  if (iter % 25 == 0) {
    // console.log('data[i]', data[i])
    // console.log('labels[i]', labels[i])
    // every 10 iterations...
    console.log(
      '%ctraining accuracy at iter ' + iter + '/500' + ': ' + evalTrainingAccuracy(),
      'color:green;font-size:1.2em',
    )
  }
}

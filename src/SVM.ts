import { Unit } from './Gate'
import { Circuit } from './Circuit'
// SVM:
export interface SVM {
  learnFrom(x: Unit, y: Unit, label: number): void
  a: Unit
  b: Unit
  c: Unit
  circuit: Circuit
  forward(x: Unit, y: Unit): Unit
  backward(): void
}
export type SVMCreator = { new (): SVM }

// SVM class -> f(x,y)=ax+by+c .
const SVM = function (this: SVM) {
  // random initial parameter values
  this.a = new Unit(1.0, 0.0)
  this.b = new Unit(-2.0, 0.0)
  this.c = new Unit(-1.0, 0.0)

  this.circuit = new Circuit()
} as unknown as SVMCreator

SVM.prototype = {
  forward: function (x: number, y: number) {
    // assume x and y are Units
    this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c)
    return this.unit_out
  },
  backward: function (label: number) {
    // label is +1 or -1

    // reset pulls on a,b,c
    this.a.grad = 0.0
    this.b.grad = 0.0
    this.c.grad = 0.0

    // compute the pull based on what the circuit output was
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
    const step_size = 0.001
    this.a.value += step_size * this.a.grad
    this.b.value += step_size * this.b.grad
    this.c.value += step_size * this.c.grad
  },
}

const data: number[][] = [],
  labels: number[] = []
data.push([1.2, 0.7])
labels.push(1)
data.push([-0.3, -0.5])
labels.push(-1)
data.push([3.0, 0.1])
labels.push(1)
data.push([-0.1, -1.0])
labels.push(-1)
data.push([-1.0, 1.1])
labels.push(-1)
data.push([2.1, -3])
labels.push(1)

console.table(data)

// const svm = new SVM()

// // a function that computes the classification accuracy
// const evalTrainingAccuracy = function () {
//   let num_correct = 0
//   for (let i = 0; i < data.length; i++) {
//     const x = new Unit(data[i][0], 0.0)
//     const y = new Unit(data[i][1], 0.0)
//     const true_label = labels[i]

//     // see if the prediction matches the provided label
//     const predicted_label = svm.forward(x, y).value > 0 ? 1 : -1
//     if (predicted_label === true_label) {
//       num_correct++
//     }
//   }
//   return num_correct / data.length
// }

// // the learning loop
// for (let iter = 0; iter < 500; iter++) {
//   // pick a random data point
//   const i = Math.floor(Math.random() * data.length)

//   const x = new Unit(data[i][0], 0.0)
//   const y = new Unit(data[i][1], 0.0)

//   const label = labels[i]

//   svm.learnFrom(x, y, label)

//   if (iter % 25 == 0) {
//     // console.log('data[i]', data[i])
//     // console.log('labels[i]', labels[i])
//     // every 10 iterations...
//     console.log('%ctraining accuracy at iter ' + iter + ': ' + evalTrainingAccuracy(), 'color:green;font-size:1.2em')
//   }
// }

var a = 1,
  b = -2,
  c = -1 // initial parameters
for (var iter = 0; iter < 400; iter++) {
  // pick a random data point
  var i = Math.floor(Math.random() * data.length)
  var x = data[i][0]
  var y = data[i][1]
  var label = labels[i]

  // compute pull
  var score = a * x + b * y + c
  var pull = 0.0
  if (label === 1 && score < 1) pull = 1
  if (label === -1 && score > -1) pull = -1

  // compute gradient and update parameters
  var step_size = 0.01
  a += step_size * (x * pull - a) // -a is from the regularization
  b += step_size * (y * pull - b) // -b is from the regularization
  c += step_size * (1 * pull)
}
console.log(a, b, c)

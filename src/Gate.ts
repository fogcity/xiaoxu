export interface Unit {
  value: number
  grad: number
}

export type UnitFactory = { new (value: number, grad: number): Unit }

export interface Gate {
  forward(u0: Unit, u1: Unit): Unit
  backward(): void
}

export interface GateConstructor {
  new (): Gate
}

/**
 * activation function gate
 */
export interface SigmoidGate {
  forward(u0: Unit): Unit
  backward(): void
  sigmoid(x: number): number
}

export interface SigmoidGateConstructor {
  new (): SigmoidGate
}

export const Unit = function (this: Unit, value: number, grad: number) {
  // value computed in the forward pass
  this.value = value
  // the derivative of circuit output w.r.t this unit, computed in backward pass
  this.grad = grad
} as unknown as UnitFactory

export const multiplyGate = function () {} as unknown as GateConstructor
multiplyGate.prototype = {
  forward: function (u0: Unit, u1: Unit) {
    // store pointers to input Units u0 and u1 and output unit utop
    this.u0 = u0
    this.u1 = u1
    this.utop = new Unit(u0.value * u1.value, 0.0)
    return this.utop
  },
  backward: function () {
    // take the gradient in output unit and chain it with the
    // local gradients, which we derived for multiply gate before
    // then write those gradients to those Units.

    this.u0.grad += this.u1.value * this.utop.grad
    this.u1.grad += this.u0.value * this.utop.grad
  },
}

export const addGate = function () {} as unknown as GateConstructor
addGate.prototype = {
  forward: function (u0: Unit, u1: Unit) {
    this.u0 = u0
    this.u1 = u1 // store pointers to input units
    this.utop = new Unit(u0.value + u1.value, 0.0)
    return this.utop
  },
  backward: function () {
    // add gate. derivative wrt both inputs is 1
    this.u0.grad += 1 * this.utop.grad
    this.u1.grad += 1 * this.utop.grad
  },
}

export const sigmoidGate = function (this: any) {
  // helper function
  this.sigmoid = function (x: number) {
    return 1 / (1 + Math.exp(-x))
  }
} as unknown as SigmoidGateConstructor
sigmoidGate.prototype = {
  forward: function (u0: Unit) {
    this.u0 = u0
    this.utop = new Unit(this.sigmoid(this.u0.value), 0.0)
    return this.utop
  },
  backward: function () {
    let s = this.sigmoid(this.u0.value)

    this.u0.grad += s * (1 - s) * this.utop.grad
  },
}

// // create input units
// let a: any = new Unit(1.0, 0.0)
// let b: any = new Unit(2.0, 0.0)
// let c: any = new Unit(-3.0, 0.0)
// let x: any = new Unit(-1.0, 0.0)
// let y: any = new Unit(3.0, 0.0)

// // create the gates
// let mulg0 = new multiplyGate()
// let mulg1 = new multiplyGate()
// let addg0 = new addGate()
// let addg1 = new addGate()
// let sg0 = new sigmoidGate()

// // do the forward pass
// const forwardNeuron = function () {
//   const ax = mulg0.forward(a, x) // a*x = -1
//   const by = mulg1.forward(b, y) // b*y = 6
//   const axpby = addg0.forward(ax, by) // a*x + b*y = 5
//   const axpbypc = addg1.forward(axpby, c) // a*x + b*y + c = 2
//   const s = sg0.forward(axpbypc) // sigmoid(a*x + b*y + c) = 0.8808

//   return s
// }

// let s = forwardNeuron()
// console.log('circuit output after one backprop: ' + s.value) // prints 0.8825

// s.grad = 1.0
// sg0.backward() // writes gradient into axpbypc
// addg1.backward() // writes gradients into axpby and c
// addg0.backward() // writes gradients into ax and by
// mulg1.backward() // writes gradients into b and y
// mulg0.backward() // writes gradients into a and x

// let step_size = 0.01
// a.value += step_size * a.grad // a.grad is -0.105
// b.value += step_size * b.grad // b.grad is 0.315
// c.value += step_size * c.grad // c.grad is 0.105
// x.value += step_size * x.grad // x.grad is 0.105
// y.value += step_size * y.grad // y.grad is 0.210

// s = forwardNeuron()
// console.log('circuit output after one backprop: ' + s.value) // prints 0.8825
// s.grad = 1.0
// sg0.backward() // writes gradient into axpbypc
// addg1.backward() // writes gradients into axpby and c
// addg0.backward() // writes gradients into ax and by
// mulg1.backward() // writes gradients into b and y
// mulg0.backward() // writes gradients into a and x
// step_size = 0.01
// a.value += step_size * a.grad // a.grad is -0.105
// b.value += step_size * b.grad // b.grad is 0.315
// c.value += step_size * c.grad // c.grad is 0.105
// x.value += step_size * x.grad // x.grad is 0.105
// y.value += step_size * y.grad // y.grad is 0.210

// s = forwardNeuron()
// console.log('circuit output after one backprop: ' + s.value) // prints 0.8825
// ;(() => {
//   const forwardCircuitFast = function (a: number, b: number, c: number, x: number, y: number) {
//     return 1 / (1 + Math.exp(-(a * x + b * y + c)))
//   }
//   a = 1
//   b = 2
//   c = -3
//   x = -1
//   y = 3
//   const h = 0.0001
//   const a_grad = (forwardCircuitFast(a + h, b, c, x, y) - forwardCircuitFast(a, b, c, x, y)) / h
//   const b_grad = (forwardCircuitFast(a, b + h, c, x, y) - forwardCircuitFast(a, b, c, x, y)) / h
//   const c_grad = (forwardCircuitFast(a, b, c + h, x, y) - forwardCircuitFast(a, b, c, x, y)) / h
//   const x_grad = (forwardCircuitFast(a, b, c, x + h, y) - forwardCircuitFast(a, b, c, x, y)) / h
//   const y_grad = (forwardCircuitFast(a, b, c, x, y + h) - forwardCircuitFast(a, b, c, x, y)) / h
// })()

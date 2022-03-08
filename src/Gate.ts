export interface Unit {
  value: number
  grad: number
}

export type UnitConstructor = { new (value: number, grad: number): Unit }

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

export interface ReLUGate {
  forward(u0: Unit): Unit
  backward(): void
  relu(x: number): number
}

export interface ReLUGateConstructor {
  new (): ReLUGate
}

export const Unit = function (this: Unit, value: number, grad: number) {
  // value computed in the forward pass
  this.value = value
  // the derivative of circuit output w.r.t this unit, computed in backward pass
  this.grad = grad
} as unknown as UnitConstructor

export const MultiplyGate = function () {} as unknown as GateConstructor
MultiplyGate.prototype = {
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

export const AddGate = function () {} as unknown as GateConstructor
AddGate.prototype = {
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

export const SigmoidGate = function (this: any) {
  // helper function
  this.sigmoid = function (x: number) {
    return 1 / (1 + Math.exp(-x))
  }
} as unknown as SigmoidGateConstructor
SigmoidGate.prototype = {
  forward: function (u0: Unit) {
    this.u0 = u0
    this.utop = new Unit(this.sigmoid(this.u0.value), 0.0)
    return this.utop
  },
  backward: function () {
    const s = this.sigmoid(this.u0.value)

    this.u0.grad += s * (1 - s) * this.utop.grad
  },
}

export const ReLUGate = function (this: any) {
  this.relu = function (x: number) {
    return Math.max(x, 0)
  }
} as unknown as SigmoidGateConstructor

ReLUGate.prototype = {
  forward: function (u0: Unit) {
    this.u0 = u0
    this.utop = new Unit(this.relu(this.u0.value), 0.0)
    return this.utop
  },
  backward: function () {
    const r = this.relu(this.u0.value)

    this.u0.grad += r > 0 ? 1.0 * this.utop.grad : 0.0
  },
}

// ax+by+c
// create input units
const a = new Unit(1.0, 0.0)
const b = new Unit(2.0, 0.0)
const c = new Unit(-3.0, 0.0)
const x = new Unit(-1.0, 0.0)
const y = new Unit(3.0, 0.0)

// create the gates
const mulg0 = new MultiplyGate()
const mulg1 = new MultiplyGate()
const addg0 = new AddGate()
const addg1 = new AddGate()
const sg0 = new SigmoidGate()

// do the forward pass
const forwardNeuron = function () {
  const ax = mulg0.forward(a, x) // a*x = -1
  const by = mulg1.forward(b, y) // b*y = 6
  const axpby = addg0.forward(ax, by) // a*x + b*y = 5
  const axpbypc = addg1.forward(axpby, c) // a*x + b*y + c = 2
  const s = sg0.forward(axpbypc) // sigmoid(a*x + b*y + c) = 0.8808
  return s
}
const sgd = (show: boolean) => {
  let s = forwardNeuron()
  if (show) console.log('%ccircuit output after forwardNeuron: ' + s.value, 'color:blue') // prints 0.8825

  s.grad = 1.0
  sg0.backward() // writes gradient into axpbypc
  addg1.backward() // writes gradients into axpby and c
  addg0.backward() // writes gradients into ax and by
  mulg1.backward() // writes gradients into b and y
  mulg0.backward() // writes gradients into a and x

  const eta = 0.0001
  a.value += eta * a.grad // a.grad is -0.105
  b.value += eta * b.grad // b.grad is 0.315
  c.value += eta * c.grad // c.grad is 0.105
  x.value += eta * x.grad // x.grad is 0.105
  y.value += eta * y.grad // y.grad is 0.210

  s = forwardNeuron()
  return s
}
const checkGradients = () => {
  const forwardCircuitFast = function (a: number, b: number, c: number, x: number, y: number) {
    return 1 / (1 + Math.exp(-(a * x + b * y + c)))
  }

  const a = 1
  const b = 2
  const c = -3
  const x = -1
  const y = 3
  const eta = 0.0001

  const a_grad = (forwardCircuitFast(a + eta, b, c, x, y) - forwardCircuitFast(a, b, c, x, y)) / eta
  const b_grad = (forwardCircuitFast(a, b + eta, c, x, y) - forwardCircuitFast(a, b, c, x, y)) / eta
  const c_grad = (forwardCircuitFast(a, b, c + eta, x, y) - forwardCircuitFast(a, b, c, x, y)) / eta
  const x_grad = (forwardCircuitFast(a, b, c, x + eta, y) - forwardCircuitFast(a, b, c, x, y)) / eta
  const y_grad = (forwardCircuitFast(a, b, c, x, y + eta) - forwardCircuitFast(a, b, c, x, y)) / eta

  console.log(`%cgradients check[${[a_grad, b_grad, c_grad, x_grad, y_grad]}]`, 'color:#d2d')
}
export const fit = (epochs: number) => {
  let w = 0
  let s;
  while (w <= epochs) {
    if (w == 1) {
      console.log(`%cgradients[${[a.grad, b.grad, c.grad, x.grad, y.grad]}]`, 'color:#d27')
      checkGradients()
    }
   s = sgd(w % 25 == 0)
    if (w % 25 == 0) {
      console.log('%ccircuit output after one backprop: ' + s.value, 'color:red')
    }

    w++
  }

  return s
}


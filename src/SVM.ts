import { Unit } from './Gate'
import { Circuit, createCircuit } from './Circuit'
// 折叶损失（hinge loss），有时候又被称为最大边界损失（max-margin loss）
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

export const SVM = function (this: SVM) {
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
    const step_size = 0.01
    this.a.value += step_size * this.a.grad
    this.b.value += step_size * this.b.grad
    this.c.value += step_size * this.c.grad
  },
}


import { MultiplyGate, AddGate, Gate } from './Gate'

export interface Circuit {
  mulg0: Gate
  mulg1: Gate
  addg0: Gate
  addg1: Gate
  forward(x: number, y: number, a: number, b: number, c: number): number
  backward(): void
}
export type CircuitConstructor = { new (): Circuit }

export const Circuit = function (this: Circuit) {
  this.mulg0 = new MultiplyGate()
  this.mulg1 = new MultiplyGate()
  this.addg0 = new AddGate()
  this.addg1 = new AddGate()
} as unknown as CircuitConstructor
Circuit.prototype = {
  forward: function (x: number, y: number, a: number, b: number, c: number) {
    this.ax = this.mulg0.forward(a, x) // a*x
    this.by = this.mulg1.forward(b, y) // b*y
    this.axpby = this.addg0.forward(this.ax, this.by) // a*x + b*y
    this.axpbypc = this.addg1.forward(this.axpby, c) // a*x + b*y + c
    return this.axpbypc
  },
  backward: function (gradient_top: number) {
    // takes pull from above
    this.axpbypc.grad = gradient_top
    this.addg1.backward() // sets gradient in axpby and c
    this.addg0.backward() // sets gradient in ax and by
    this.mulg1.backward() // sets gradient in b and y
    this.mulg0.backward() // sets gradient in a and x
  },
}

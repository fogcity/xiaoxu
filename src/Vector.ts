import { Point } from './Point'

export const Vector = class<T> {
  private v: T[] = []
  public head: typeof Point | undefined
  public tail: typeof Point | undefined
  constructor(n?: T | T[]) {
    if (n) {
      if (n instanceof Array) {
        this.v = n
      } else {
        this.v = new Array(n)
      }
    }
  }

  [n: number]: T
  toArray() {
    return this.v
  }
  get x() {
    if (this.v.length >= 1) return this.v[0]
    throw new Error("it isn't a vector!")
  }
  set x(n: T) {
    this.v[0] = n
  }
  get y() {
    if (this.v.length >= 2) return this.v[1]
    throw new Error("it isn't a 2D-vector!")
  }
  get z() {
    if (this.v.length >= 3) return this.v[2]
    throw new Error("it isn't a 3D-vector!")
  }
  get w() {
    if (this.v.length >= 4) return this.v[3]
    throw new Error("it isn't a 4D-vector!")
  }

  static ones(n: number) {
    return new Vector(n).fill(1)
  }

  static zeros(n: number) {
    return new Vector(n).fill(0)
  }

  fill(n: T) {
    this.v.fill(n)
    return this
  }
}

const v = new Vector([1, 2, 3])
const v3 = Vector.ones(3)

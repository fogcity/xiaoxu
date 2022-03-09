import { Vector } from './Vector'

export const Point = class<T> {
  private v: T[]

  constructor(n: T | T[]) {
    if (n instanceof Array) {
      this.v = n
    } else {
      this.v = new Array(n)
    }
  }

  [n: number]: T
  toArray() {
    return this.v
  }
  sub(t: typeof Point | typeof Vector) {
    if (typeof t == typeof Vector) {
      //   const r = new Point([this.x - t.x])
    } else if (typeof t == typeof Point) {
      const r = new Vector()
      r.head = this as unknown as typeof Point
      r.tail = t as typeof Point
      return
    }
  }

  get x() {
    if (this.v.length >= 1) return this.v[0]
    throw new Error("it isn't a vector!")
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
    return new Point(n).fill(1)
  }

  static zeros(n: number) {
    return new Point(n).fill(0)
  }

  fill(n: T) {
    this.v.fill(n)
    return this
  }
}

const v = new Point([1, 2, 3])
const v3 = Point.ones(3)

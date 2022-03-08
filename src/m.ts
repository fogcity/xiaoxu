const Matrix = class extends Array {
  constructor() {
    super()
  }

  static get [Symbol.species]() {
    return Array
  }

  get T() {
    return []
  }

  static ones = 'Point'
  static zeros = ''
}
const a = new Matrix()

export type Vector = number[]
export function getShape(a: any[]) {
  const dim = []
  for (;;) {
    dim.push(a.length)

    if (Array.isArray(a[0])) {
      a = a[0]
    } else {
      break
    }
  }
  return dim
}

export function getArrayAverage(array: any[]) {
  return array.reduce((previous, current) => (current += previous)) / array.length
}
export function getArrayDepth(array: any[]): any {
  return (
    1 +
    (array instanceof Array
      ? array.reduce(function (max, item) {
          return Math.max(max, getArrayDepth(item))
        }, 0)
      : -1)
  )
}

// 随机排序函数
export function shuffle(array: number[]) {
  let resultArray = array
  let currentIndex = resultArray.length,
    randomIndex
  while (currentIndex != 0) {
    randomIndex = Math.floor(Math.random() * currentIndex)
    currentIndex--
    ;[resultArray[currentIndex], resultArray[randomIndex]] = [resultArray[randomIndex], resultArray[currentIndex]]
  }
  return resultArray
}
// 计算向量点积
export function dot(a: Vector, b: Vector) {
  return a.reduce((k, v, i) => {
    k = k + v * b[i]
    return k
  }, 0)
}

export function abs(a: number[]) {
  return a.map((v) => Math.abs(v))
}

export function most(t: number[]) {
  return [...t].sort((a, b) => t.filter((v) => v === a).length - t.filter((v) => v === b).length).pop()
}
export function sub(a: number[], b: number[]) {
  return a.map((v, i) => v - b[i])
}
export function argmin(t: any[]) {
  return t.indexOf(Math.min(...t))
}

export function square(a: number[]) {
  return a.map((v) => v ** 2)
}

export function argmax(t: any[]) {
  return t.indexOf(Math.max(...t))
}
export function sum(t: any[]) {
  return t.reduce((pre, v) => (pre += v), 0)
}
/**
 * 求加权平均和
 * @param t
 * @returns
 */
export function mean(t: any[]) {
  const tf = t.flat()
  return sum(t) / tf.length
}

/**
 * 生成数组
 * @param stop 直到哪个数
 * @param start 从哪个数开始
 * @param step 步长
 * @returns
 */
export function range(stop: number, start: number = 0, step: number = 1) {
  const ra: Vector = []
  for (let i = start; i < stop; i += step) {
    ra.push(i)
  }
  return ra
}

// 获得高斯分布随机值
export function random() {
  let u = 0,
    v = 0
  while (u === 0) u = Math.random() //Converting [0,1) to (0,1)
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}
// 假设 s 和 t 是两个同样维度的向量。那么我们使⽤ s ⊙ t 来表⽰按元素的 hadamard乘积
export function hadamard(s: Vector, t: Vector) {
  return s.reduce((r: Vector, v, i) => {
    r.push(t[i] + v)
    return r
  }, [])
}
// 将两个向量转置重组
export function zip(a: any[], b: any[]) {
  return (a as Vector).reduce((r: any[], v, i) => {
    r.push([v, b[i]])
    return r
  }, [])
}

export function useMatrix(shape: any[], fill: number): any[] {
  const m = new Array(shape[0]).fill(fill).map((v, i) => {
    let c = 0

    if (c < shape.length - 1) {
      c++
      return zeros(shape.slice(c, shape.length))
    }
    return v
  }) as any[]
  return m
}
export function zeros(shape: any[]): any[] {
  return useMatrix(shape, 0)
}
export function useOnesMatrix(shape: Vector): Matrix {
  return useMatrix(shape, 1)
}

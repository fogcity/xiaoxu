export type Vector = number[]
export type Matrix = number[] | number[][] | number[][][] | number[][][][]
export function getShape(v: Matrix) {
  return [v.length, getArrayDepth(v)]
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
export function useShuffleArray(array: number[]) {
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

/**
 * 生成数组
 * @param stop 直到哪个数
 * @param start 从哪个数开始
 * @param step 步长
 * @returns
 */
export function useRange(stop: number, start: number = 0, step: number = 1) {
  const ra: Vector = []
  for (let i = start; i < stop; i += step) {
    ra.push(i)
  }
  return ra
}
// 获得高斯分布随机值
export function useRandn() {
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
export function useZip(a: Matrix, b: Matrix) {
  return (a as Vector).reduce((r: any[], v, i) => {
    r.push([v, b[i]])
    return r
  }, [])
}
export function useMatrix(shape: Matrix, fill: number): Matrix {
  const m = new Array(shape[0]).fill(fill).map((v, i) => {
    let c = 0

    if (c < shape.length - 1) {
      c++
      return useZerosMatrix(shape.slice(c, shape.length))
    }
    return v
  }) as Matrix
  return m
}
export function useZerosMatrix(shape: Matrix): Matrix {
  return useMatrix(shape, 0)
}
export function useOnesMatrix(shape: Vector): Matrix {
  return useMatrix(shape, 1)
}

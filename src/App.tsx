import { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
const zip = (a: number[], b: number[]) => a.map((k, i) => [k, b[i]])
const randn_bm = () => {
  var u = 0,
    v = 0
  while (u === 0) u = Math.random() //Converting [0,1) to (0,1)
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}
const sigmoid = (z: number) => 1.0 / (1.0 + Math.exp(-z))
function network(sizes: number[]) {
  const b = sizes.slice(1).map((v) => {
    let a = []
    let b = []
    for (let index = 0; index < v; index++) {
      b.push(randn_bm())
    }
    a.push(b)
    return a
  })

  const w = zip(sizes.slice(0, -1), sizes.slice(1)).map(([x, y]) => {
    let a = []
    let b = []
    for (let index = 0; index < y; index++) {
      for (let index = 0; index < x; index++) {
        b.push(randn_bm())
      }
      a.push(b)
      b = []
    }
    return a
  })

  const n = {
    num_layers: sizes.length,
    sizes,
    biases: b,
    weights: w,
  }
  return n
}
function App() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    console.log(network([2, 3, 1]))
  })

  return <div></div>
}

export default App

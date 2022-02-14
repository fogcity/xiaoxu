import { useEffect, useState } from 'react'
import { getShape, getArrayDepth, Vector, Matrix, useZerosMatrix, shuffle, xrange, zip, randn_bm, dot } from './m'

type Network = {
  num_layers: number // 层数
  sizes: Vector // 尺寸
  biases: Matrix // 偏差
  weights: Matrix // 权重
}

// s神经元激活函数
function sigmoid(z: number, k: number = 1) {
  // if (z instanceof Array) {
  //   return z.map((v) => 1 / (1 + Math.exp(-v / k))) as Vector
  // }
  return 1 / (1 + Math.exp(-z / k))
}

// s神经元激活函数的导数
function sigmoid_derivative(z: number) {
  // if (z instanceof Array) {
  //   const s = (v: number) => sigmoid(v) as number
  //   return z.map((v) => s(v) * (1 - s(v)))
  // }
  const s = sigmoid(z) as number
  return s * (1 - s)
}

function cost_derivative(output_activations: number, y: number) {
  return output_activations - y
}

function vectorized_result(shape: Vector, target: number) {
  const e = useZerosMatrix(shape)
  e[target] = 1.0
  return e
}

// 获得前馈神经网络的值
const feedforward = (n: Network, a: Vector) => {
  return zip(n.biases, n.weights).map(([b, w]: [number, Vector]) => sigmoid(dot(w, a) + b))
}

/**
 * 随机梯度下降训练
 * @param training_data ⼀个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出
 * @param epochs 迭代期数量
 * @param mini_batch_size 采样时的⼩批量数据的⼤⼩
 * @param eta 学习率
 * @param test_data 测试集
 */
const SGD = (
  network: Network,
  training_data: Vector,
  epochs: number,
  mini_batch_size: number,
  eta: number,
  test_data?: Vector,
) => {
  const n_test = test_data?.length // 测试集数量
  const n = training_data.length // 训练集数量
  for (let j = 0; j < epochs; j++) {
    shuffle(training_data) // 随机打乱训练集的顺序

    const mini_batches = []
    for (const k of xrange(n, 0, mini_batch_size)) {
      mini_batches.push(training_data.slice(k, k + mini_batch_size))
    }

    for (const mini_batch of mini_batches) {
      update_mini_batch(network, mini_batch, eta)
    }

    if (test_data) {
      console.log(`Epoch ${j}: ${evaluate(test_data)} / ${n_test}`)
    } else console.log(`Epoch ${j} complete`)
  }
}
/**
 * 通过使用反向传播来应用梯度下降来更新网络的权重和偏差。
 * @param mini_batch  (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出
 * @param eta 学习率
 */
const update_mini_batch = (n: Network, mini_batch: Matrix, eta: number) => {
  let nabla_b = n.biases.map((b) => {
    return useZerosMatrix(getShape(b as Matrix))
  }) as Matrix

  let nabla_w = n.weights.map((w) => {
    return useZerosMatrix(getShape(w as Matrix))
  }) as Matrix
  for (const batch of mini_batch) {
    const [x, y] = batch as number[]
    //调⽤了⼀个称为`反向传播backprop`的算法,⼀种快速计算代价函数的梯度的⽅法。因此
    // update_mini_batch 的⼯作仅仅是对 mini_batch 中的每⼀个训练样本计算梯度，然后适当地更新 self.weights 和 self.biases
    const [delta_nabla_b, delta_nabla_w] = backprop(n, x, y)

    nabla_b = zip(nabla_b, delta_nabla_b).map(([nb, dnb]) => nb + dnb)
    nabla_w = zip(nabla_w, delta_nabla_w).map(([nw, dnw]) => nw + dnw)

    n.weights = zip(n.weights, nabla_w).map(([w, nw]) => w - (eta / mini_batch.length) * nw)
    n.biases = zip(n.biases, nabla_b).map(([b, nb]) => b - (eta / mini_batch.length) * nb)
    return n
  }
}

function backprop(n: Network, x: Vector, y: Vector) {
  let nabla_b = n.biases.map((b) => useZerosMatrix(b as Matrix))
  let nabla_w = n.weights.map((w) => useZerosMatrix(w as Matrix))
  let activation: number | Vector = x
  let activations = [x] // list to store all the activations, layer by layer

  let zs = [] // list to store all the z vectors, layer by layer

  zip(n.biases, n.weights).map(([b, w]: [number, Vector]) => {
    const z = dot(w, activation as Vector) + b
    zs.push(z)
    activation = sigmoid(z)
    activations.push(activation as Vector)
  })
  // backward pass
  // const delta = cost_derivative(activations[activations.length - 1], y) * sigmoid_derivative(zs[zs.length - 1])
  // nabla_b[-l] = delta
  // nabla_w[-l] = dot(delta, activations[-l - 1].transpose())
  // return nabla_b, nabla_w
}
function network(sizes: Vector) {
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

  const n: Network = {
    num_layers: sizes.length,
    sizes,
    biases: b,
    weights: w,
  }

  return n
}

const evaluate = (test_data: Vector) => {}

function App() {
  const [count, setCount] = useState(0)

  useEffect(() => {})

  return <div></div>
}

export default App

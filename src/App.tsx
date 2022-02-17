import { useEffect, useState } from 'react'
import {
  getShape,
  getArrayDepth,
  Vector,
  Matrix,
  useZerosMatrix,
  useShuffleArray,
  useRange,
  useZip,
  useRandn,
  dot,
} from './m'
import {} from './SVM'
type Network = {
  num_layers: number // 层数
  sizes: Vector // 尺寸
  biases: Matrix // 偏差
  weights: Matrix // 权重
}

// s神经元激活函数
function sigmoid(z: number, k: number = 1) {
  return 1 / (1 + Math.exp(-z / k))
}

// s神经元激活函数的导数
function sigmoid_derivative(z: number) {
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
const useFeedforward = (n: Network, a: Vector) => {
  return useZip(n.biases, n.weights).map(([b, w]: [number, Vector]) => sigmoid(dot(w, a) + b))
}

/**
 * 随机梯度下降训练
 * @param training_data ⼀个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出
 * @param epochs 迭代期数量
 * @param mini_batch_size 采样时的⼩批量数据的⼤⼩
 * @param eta 学习率
 * @param test_data 测试集
 */
const useSgd = (
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
    const td = useShuffleArray(training_data)

    const mini_batches = []
    for (const k of useRange(n, 0, mini_batch_size)) {
      mini_batches.push(td.slice(k, k + mini_batch_size))
    }

    for (const mini_batch of mini_batches) {
      updateMiniBatch(network, mini_batch, eta)
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
const updateMiniBatch = (n: Network, mini_batch: Matrix, eta: number) => {
  let nabla_b = n.biases.map((b) => {
    return useZerosMatrix(getShape(b as Matrix))
  }) as Matrix

  let nabla_w = n.weights.map((w) => {
    return useZerosMatrix(getShape(w as Matrix))
  }) as Matrix
  for (const batch of mini_batch) {
    const [x, y] = batch as number[]

    const [delta_nabla_b, delta_nabla_w] = useBackprop(n, x, y)

    nabla_b = useZip(nabla_b, delta_nabla_b).map(([nb, dnb]) => nb + dnb)
    nabla_w = useZip(nabla_w, delta_nabla_w).map(([nw, dnw]) => nw + dnw)

    n.weights = useZip(n.weights, nabla_w).map(([w, nw]) => w - (eta / mini_batch.length) * nw)
    n.biases = useZip(n.biases, nabla_b).map(([b, nb]) => b - (eta / mini_batch.length) * nb)
    return n
  }
}

function useBackprop(n: Network, x: Vector, y: Vector) {
  let nabla_b = n.biases.map((b) => useZerosMatrix(b as Matrix))
  let nabla_w = n.weights.map((w) => useZerosMatrix(w as Matrix))
  let activation: number | Vector = x
  let activations = [x] // list to store all the activations, layer by layer

  let zs = [] // list to store all the z vectors, layer by layer

  useZip(n.biases, n.weights).map(([b, w]: [number, Vector]) => {
    const z = dot(w, activation as Vector) + b
    zs.push(z)
    activation = sigmoid(z)
    activations.push(activation as unknown as Vector)
  })
  // backward pass
  // const delta = cost_derivative(activations[activations.length - 1], y) * sigmoid_derivative(zs[zs.length - 1])
  // nabla_b[-l] = delta
  // nabla_w[-l] = dot(delta, activations[-l - 1].transpose())
  // return nabla_b, nabla_w
}

// mean absolute error
function useMean() {}
function useNetwork(sizes: number[]) {
  const size_b = sizes.slice(1)
  const size_w = sizes.slice(0, -1)
  const b = size_b.map((v) => {
    let a = []
    let b = []
    for (let index = 0; index < v; index++) {
      b.push(useRandn())
    }
    a.push(b)
    return a
  })

  const w = useZip(size_w, size_b).map(([x, y]) => {
    let a = []
    let b = []
    for (let index = 0; index < y; index++) {
      for (let index = 0; index < x; index++) {
        b.push(useRandn())
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
  const [trainData, setTrainData] = useState({
    sizeMB: [
      0.08, 9.0, 0.001, 0.1, 8.0, 5.0, 0.1, 6.0, 0.05, 0.5, 0.002, 2.0, 0.005, 10.0, 0.01, 7.0, 6.0, 5.0, 1.0, 1.0,
    ],
    timeSec: [
      0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116, 0.07, 0.289, 0.076, 0.744, 0.083, 0.56,
      0.48, 0.399, 0.153, 0.149,
    ],
  })

  const [testData, setTestData] = useState({
    sizeMB: [
      5.0, 0.2, 0.001, 9.0, 0.002, 0.02, 0.008, 4.0, 0.001, 1.0, 0.005, 0.08, 0.8, 0.2, 0.05, 7.0, 0.005, 0.002, 8.0,
      0.008,
    ],
    timeSec: [
      0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.07, 0.375, 0.058, 0.136, 0.052, 0.063, 0.183, 0.087, 0.066, 0.558,
      0.066, 0.068, 0.61, 0.057,
    ],
  })
  // chain rule : ∂f(q,z)∂x=∂q(x,y)∂x∂f(q,z)∂q
  useEffect(() => {})

  return <div></div>
}

export default App

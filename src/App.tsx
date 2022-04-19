import { useCallback, useEffect, useState, version } from 'react'
import { fit, Unit } from './Gate'
import { KNearestNeighbor } from './KNN'
import * as tf from '@tensorflow/tfjs'
import { mean, shuffle, sum, zip } from './m'
import { SVM } from './SVM'
import { svmLoss } from './SVM2d'
import { NN } from './nn'
import * as nn from './NN-based'
import { randomNormal, tensor, tidy } from '@tensorflow/tfjs'
import { randomReLUWeight } from './weight'
import { log } from './utils'
import { vis } from '../../vis/example/build'
import * as d3 from 'd3'
export const loadData = async (trainSize: number = 50, testSize: number = 50) => {
  const dataCsvConfigs = {
    columnConfigs: {
      label: {
        isLabel: true,
      },
    },
  }

  const trainDataAll = tf.data.csv('/data/fashion-mnist_train.csv', dataCsvConfigs)
  const testDataAll = tf.data.csv('/data/fashion-mnist_test.csv', dataCsvConfigs)

  const trainData = await trainDataAll
    .shuffle(3)
    .take(trainSize)
    .map((res: any) => {
      return { xs: Object.values(res.xs), ys: Object.values(res.ys) }
    })
    .toArray()

  const testData = await testDataAll
    .shuffle(3)
    .take(testSize)
    .map((res: any) => {
      return { xs: Object.values(res.xs), ys: Object.values(res.ys) }
    })
    .toArray()

  const numOfFeatures = (await testDataAll.columnNames()).length - 1
  return { trainData, testData, numOfFeatures }
}

const knnTrain = async function () {
  try {
    const { trainData, testData, numOfFeatures } = await loadData(10, 100)
    const Xtr = tensor(trainData.map((v) => v.xs) as number[][])
    const Ytr = tensor(trainData.map((v) => v.ys) as number[][])
    const Xte = tensor(testData.map((v) => v.xs) as number[][])
    const Yte = tensor(testData.map((v) => v.ys) as number[][])

    const knn = new KNearestNeighbor('L2')
    knn.train(Xtr, Ytr)
    const Yte_pre = knn.predict(Xte, 7)
    tidy(() => {
      console.log(
        `accuracy: ${
          sum(
            zip(Yte_pre.flat(), (Yte.arraySync() as number[]).flat()).map((v, i) => {
              if (v[0] == v[1]) {
                return 100
              }

              return 0
            }),
          ) / (Yte_pre as number[]).length
        }%`,
      )
    })
  } catch (e) {
    console.error(e)
  }
}

const nnTrain = async function () {
  try {
    // const { trainData, testData, numOfFeatures } = await loadData(10, 100)
    // const trainX = tensor(trainData.map((v) => v.xs) as number[][])
    // const trainY = tensor(trainData.map((v) => v.ys) as number[][])
    // const Xte = tensor(testData.map((v) => v.xs) as number[][])
    // const Yte = tensor(testData.map((v) => v.ys) as number[][])
    // const td = trainData
    // const n = new NN([2, 3, 1])
    // const d = [
    //   { xs: [[3], [2]], ys: [1] },
    //   { xs: [[4], [4]], ys: [0] },
    // ]
    const featureNumber = 4
    const dataNumber = 100
    let labels: number[] = []
    let datas: number[][] = []
    let dd = [1, 2, 3, 4]
    for (let i = 0; i < dataNumber; i++) {
      const d = [Math.random(), Math.random() - 0.5, Math.random(), Math.random()]
      if (i == 3) {
        dd = d
      }
      datas.push(d)
      labels.push(Math.random())
    }
    console.log('datas', datas)
    console.log('labels', labels)

    // const datas2 = randomNormal([dataNumber, featureNumber]).arraySync() as number[][]
    // const labels2 = randomNormal([dataNumber, 1]).arraySync() as number[]

    // n.sgd(d, 1, 1, 0.001)
    const n = nn.buildNetwork([featureNumber, 5, 5, 1], nn.Activations.LINEAR)
    const learningRate = 0.001
    const regularizationRate = 0.0001
    const batchSize = 5
    const epoch = 20
    log('start train:', 'green')
    const lossPoints: [number, number][] = []
    for (let i = 0; i < epoch; i++) {
      datas.forEach((input, i) => {
        const output = nn.forwardProp(n, input)

        nn.backProp(n, labels[i], nn.Errors.MAE)
        if ((i + 1) % batchSize == 0) nn.updateWeights(n, learningRate, regularizationRate)
      })
      const loss = nn.getLoss(n, datas, labels)
      lossPoints.push([i, loss])

      // const loss2 = nn.getLoss(n, datas2, labels2)
      // log(loss2 + '')
    }
    log('over train:', 'green')
    vis.renderLineChart(
      document.getElementById('c1') as HTMLElement,
      {
        dataset: lossPoints,
        series: [],
      },
      {
        xLabel: 'epoch',
        yLabel: 'loss',
        marginLeft: 70,
        marginBottom: 70,
        height: 300,
        yDomain: [0, Math.max(1, ...(d3.extent(lossPoints, (d) => d[1]) as number[]))],
        showXAxisGrid: true,
        showYAxisGrid: true,
        color: '#be123c',
        lineWidth: 2,
        curve: d3.curveLinear,
        xAccessor: (d: any[]) => d[0],
        yAccessor: (d: any[]) => d[1],
      },
    )
    log('predict')

    console.log(labels[3], nn.predict(n, dd))
  } catch (e) {
    console.error(e)
  }
}

const svmUnitTrain = async function () {
  try {
    const { trainData, testData, numOfFeatures } = await loadData(10, 4)

    // a function that computes the classification accuracy
    const evalTrainingAccuracy = () => {
      let num_correct = 0
      for (let i = 0; i < data.length; i++) {
        const x = new Unit(data[i][0], 0.0)
        const y = new Unit(data[i][1], 0.0)
        const true_label = labels[i]

        // see if the prediction matches the provided label
        const v = svm.forward(x, y).value

        const predicted_label = v > 0 ? 1 : -1

        if (predicted_label === true_label) {
          num_correct++
        }
      }

      return num_correct / data.length
    }

    const data: number[][] = [],
      labels: number[] = [-1, -1, 1, -1]
    data.push([1.2, 0.7])
    data.push([1.6, 0.1])
    data.push([-0.3, -0.5])
    data.push([-0.7, -0.1])

    console.table(data)

    // 初始化svm
    const svm = new SVM()

    // the learning loop
    for (let iter = 0; iter < 200000; iter++) {
      // pick a random data point
      const i = Math.floor(Math.random() * data.length)

      const x = new Unit(data[i][0], 0.0)
      const y = new Unit(data[i][1], 0.0)

      const label = labels[i]

      svm.learnFrom(x, y, label)

      if (iter % 25 == 0) {
        // console.log('data[i]', data[i])
        // console.log('labels[i]', labels[i])
        // every 10 iterations...
        console.log(
          '%ctraining accuracy at iter ' + iter + '/500' + ': ' + evalTrainingAccuracy(),
          'color:green;font-size:1.2em',
        )
      }
    }
  } catch (e) {
    console.error(e)
  }
}

const svmTrain = async () => {
  const trainNumber = 2
  const classNumber = 784
  const labelNumber = 10
  const { trainData, testData, numOfFeatures } = await loadData(trainNumber, 1)
  const xTrain = tensor(trainData.map((v) => v.xs) as number[][])

  const yTrain = tensor(trainData.map((v) => v.ys) as number[][])
  const xTest = tensor(testData.map((v) => v.xs) as number[][])
  const yTest = tensor(testData.map((v) => v.ys) as number[][])
  const w = randomReLUWeight([classNumber, labelNumber]) // example: random numbers
  const reg = 0 // regularization strength

  const [loss, grad] = await svmLoss(w, xTrain, yTrain, reg)
}
function App() {
  // chain rule : ∂f(q,z)∂x=∂q(x,y)∂x∂f(q,z)∂q
  useEffect(() => {
    // knnTrain()
    nnTrain()
  })

  return (
    <>
      <div id="c1"></div>
      <div id="c2"></div>
    </>
  )
}

export default App

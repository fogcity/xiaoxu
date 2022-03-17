import { useCallback, useEffect, useState } from 'react'
import { fit, Unit } from './Gate'
import { KNearestNeighbor } from './KNN'
import * as tf from '@tensorflow/tfjs'
import { mean, sum, zip } from './m'
import { SVM } from './SVM'
import { svmLoss } from './SVM2d'
import { tensor } from '@tensorflow/tfjs'
import { randomReLUWeight } from './weight'

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
    const { trainData, testData, numOfFeatures } = await loadData(100, 100)
    const Xtr = tensor(trainData.map((v) => v.xs) as number[][])
    const Ytr = tensor(trainData.map((v) => v.ys) as number[][])
    const Xte = tensor(testData.map((v) => v.xs) as number[][])
    const Yte = tensor(testData.map((v) => v.ys) as number[][])

    const knn = new KNearestNeighbor('L2')
    knn.train(Xtr, Ytr)
    const Yte_pre = knn.predict(Xte, 7)
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

  xTrain.shape
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
    svmTrain()
  })

  return <div></div>
}

export default App

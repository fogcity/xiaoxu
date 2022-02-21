import { useCallback, useEffect, useState } from 'react'

import * as tf from '@tensorflow/tfjs'
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
    .take(trainSize)
    .map((res: any) => {
      return { xs: Object.values(res.xs), ys: Object.values(res.ys) }
    })
    .toArray()

  const testData = await testDataAll
    .take(testSize)
    .map((res: any) => {
      return { xs: Object.values(res.xs), ys: Object.values(res.ys) }
    })
    .toArray()

  const numOfFeatures = (await testDataAll.columnNames()).length - 1
  console.log('%ctrainData', trainData)
  console.log('%ctestData', testData)
  console.log('numOfFeatures', numOfFeatures)
  return { trainData, testData, numOfFeatures }
}
function App() {
  // chain rule : ∂f(q,z)∂x=∂q(x,y)∂x∂f(q,z)∂q
  useEffect(() => {
    ;(async function () {
      try {
        const { trainData, testData, numOfFeatures } = await loadData()
      } catch (e) {
        console.error(e)
      }
    })()
  })

  return <div></div>
}

export default App

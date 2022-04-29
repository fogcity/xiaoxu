// L=[∑_i=1Nmax(0,−y_i(w_0x_i0+w_1x_i1+w_2)+1)]+α[w_02+w_12]
// In this exercise you will:

// - implement a fully-vectorized **loss function** for the SVM
// - implement the fully-vectorized expression for its **analytic gradient**
// - **check your implementation** using numerical gradient
// - use a validation set to **tune the learning rate and regularization** strength
// - **optimize** the loss function with **SGD**
// - **visualize** the final learned weights

import { Tensor, tensor, scalar, zeros, range, dot, max, Log } from '@tensorflow/tfjs'
import { loadData } from './App'
import { L2regularization } from './Regularization'
import { randomReLUWeight } from './weight'

/**
 * Structured SVM loss function, naive implementation (with loops).
 * @param w A numpy array of shape (D, C) containing weights.
 * @param x A numpy array of shape (N, D) containing a minibatch of data.
 * @param y A numpy array of shape (N,) containing training labels; y[i] = c means
 * @param reg (float) regularization strength
 */
export async function svmLoss(w: Tensor, x: Tensor, y: Tensor, reg: number) {
  const dW = zeros(w.shape) // initialize the gradient as zero
  const wArray = (await w.array()) as number[][] // (2, 784)
  const xArray = (await x.array()) as number[][] // (784, 10)
  const yArray = (await y.array()) as number[] // (10, 1)

  // compute the loss and the gradient

  const classNumber = w.shape[1] as number // 10
  const trainNumber = x.shape[0] // 2

  let loss = 0.0
  for (const i of await range(0, trainNumber).array()) {
    const scores = dot(xArray[i], wArray)

    const scoresArray = (await scores.array()) as number[]
    console.log('scoresArray', scoresArray)

    for (const j of await range(0, classNumber).array()) {
      const yi = yArray[i]
      if (j != yi) {
        const margin = scoresArray[j] - scoresArray[yi] + 1
        loss += Math.max(margin, 0)

        // compute grad
        // 公式2
        // dW[:,y[i]] += -X[i,:].T
        // 公式1
        // dW[:,j] += X[i, :].T
      }
    }
  }

  // Right now the loss is a sum over all training examples, but we want it
  // to be an average instead so we divide by numTrain.
  loss /= trainNumber

  // Add regularization to the loss.

  const regularization = await L2regularization(w, reg).data()

  loss += regularization[0]
  console.log('loss', loss)

  // compute dw

  return [loss, dW]
}

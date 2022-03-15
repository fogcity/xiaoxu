// L=[∑_i=1Nmax(0,−y_i(w_0x_i0+w_1x_i1+w_2)+1)]+α[w_02+w_12]
// In this exercise you will:

// - implement a fully-vectorized **loss function** for the SVM
// - implement the fully-vectorized expression for its **analytic gradient**
// - **check your implementation** using numerical gradient
// - use a validation set to **tune the learning rate and regularization** strength
// - **optimize** the loss function with **SGD**
// - **visualize** the final learned weights

import { Tensor, tensor, scalar, zeros, range, dot } from '@tensorflow/tfjs'
import { loadData } from './App'
import { randomReLUWeight } from './weight'

/**
 * Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Returns a tuple [loss , grad] of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
 * @param w A numpy array of shape (D, C) containing weights.
 * @param x A numpy array of shape (N, D) containing a minibatch of data.
 * @param y A numpy array of shape (N,) containing training labels; y[i] = c means
 * @param reg (float) regularization strength
 */
export async function svmLoss(w: Tensor, x: Tensor, y: Tensor, reg: number) {
  const dW = zeros(w.shape) // initialize the gradient as zero

  const wA = (await w.array()) as number[][]
  const xA = (await x.array()) as number[][]
  const yA = (await y.array()) as number[]

  // compute the loss and the gradient

  const nF = w.shape[0] as number
  console.log('numberFeature', nF)

  const nT = x.shape[0]
  console.log('numTrain', nT)

  let loss = 0.0
  for (const i of await range(0, nT).array()) {
    // map the input

    const scores = dot(wA, xA[i])
    const sA = (await scores.array()) as number[]
    console.log('score', sA)

    const yi = yA[i]
    console.log('y i', yi)

    const ccs = sA[i]
    console.log('correct class score', ccs)

    for (const j of await range(0, nF).array()) {
      if (j != yi) {
        const margin = sA[j] - ccs + 1
        if (margin > 0) loss += margin
      }
    }
  }
  // Right now the loss is a sum over all training examples, but we want it
  // to be an average instead so we divide by numTrain.
  loss /= nT

  // Add regularization to the loss.

  const regularization = await w.mul(w).sum().mul(scalar(reg)).data()

  loss += regularization[0]
  console.log('loss', loss)

  // compute dw

  return [loss, dW]
}

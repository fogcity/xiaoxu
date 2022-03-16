import { Tensor, max, tensor, scalar, zeros, range, dot, TensorLike } from '@tensorflow/tfjs'
import { svmLoss } from './SVM2d'
import { randomReLUWeight } from './weight'

export class LinearClassifier {
  w: Tensor | undefined
  constructor() {}
  /**
   *
   * @param x A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
   * @param y A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
   * @param learning_rate learning_rate: (float) learning rate for optimization.
   * @param reg regularization strength.
   * @param num_iters number of steps to take when optimizing
   * @param batch_size number of training examples to use at each step.
   * @param verbose If true, print progress during optimization.
   * 
   * @returns A list containing the value of the loss function at each training iteration.
   * 
   */
  async train(
    x: Tensor,
    y: Tensor,
    learning_rate = 1e-3,
    reg = 1e-5,
    num_iters = 100,
    batch_size = 200,
    verbose = false,
  ) {
    const [num_train, dim] = x.shape
    const num_classes = max(y).add(1).dataSync()[0]

    if (!this.w)
      // lazily initialize W
      this.w = randomReLUWeight([dim, num_classes])

    // Run stochastic gradient descent to optimize W
    const loss_history = []
    for (const i of range(0, num_iters).arraySync()) {
      const [loss, grad] = await this.loss(x, y, reg)
      loss_history.push(loss)
      if (verbose && i % 100 == 0) console.log(`iteration ${i}/${num_iters}: ${loss} %`)
    }
  }
  /**
   * Use the trained weights of this linear classifier to predict labels fo data points.
   * @param x A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
     @return y_pred Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
   */
  predict(x: Tensor) {
    const y_pred = zeros([x.shape[0]])

    return y_pred
  }

  /**
   * Compute the loss function and its derivative.
   * @param x_batch A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
   * @param y_batch A numpy array of shape (N,) containing labels for the minibatch.
   * @param type cost function
   * @param reg regularization strength.
   * 
   * @returns A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
   */
  async loss(x_batch: Tensor, y_batch: Tensor, reg: number, type: 'svm' | 'softmax' = 'svm') {
    switch (type) {
      case 'svm':
        return await svmLoss(this.w!, x_batch, y_batch, reg)
      //   case 'softmax':
      //   return softmax_loss_vectorized(this.w, x_batch, y_batch, reg)

      default:
        return await svmLoss(this.w!, x_batch, y_batch, reg)
        break
    }
  }
}

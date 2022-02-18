// L=[∑_i=1Nmax(0,−y_i(w_0x_i0+w_1x_i1+w_2)+1)]+α[w_02+w_12]

const X = [
  [1.2, 0.7],
  [-0.3, 0.5],
  [3, 2.5],
] // array of 2-dimensional data
const y = [1, -1, 1] // array of labels
const w = [0.1, 0.2, 0.3] // example: random numbers
const alpha = 0.1 // regularization strength

export function cost(X: string | any[], y: any[], w: number[]) {
  let total_cost = 0.0 // L, in SVM loss function above
  const N = X.length
  for (let i = 0; i < N; i++) {
    // loop over all data points and compute their score
    const xi = X[i]
    const score = w[0] * xi[0] + w[1] * xi[1] + w[2]

    // accumulate cost based on how compatible the score is with the label
    const yi = y[i] // label
    const costi = Math.max(0, -yi * score + 1)
    console.log('example ' + i + ': xi = (' + xi + ') and label = ' + yi)
    console.log('%c  score computed to be ' + score.toFixed(3), 'color:blue')
    console.log('%c  => cost computed to be ' + costi.toFixed(3), 'color:red')
    total_cost += costi
  }

  // regularization cost: we want small weights
  const reg_cost = alpha * (w[0] * w[0] + w[1] * w[1])
  console.log('%cregularization cost for current model is ' + reg_cost.toFixed(3), 'font-size:1.2em;color:#f0d')
  total_cost += reg_cost

  console.log('%ctotal cost is ' + total_cost.toFixed(3), 'font-size:1.2em;color:#db3')
  return total_cost
}
cost(X, y, w)

import { randomNormal, div, sqrt } from '@tensorflow/tfjs'

export function randomReLUWeight(shape: number[], factor = 0.001) {
  const r = sqrt(div(2.0, shape[0])).mul(factor)
  const w = randomNormal(shape)
  return w.mul(r)
}

import { scalar, Tensor } from '@tensorflow/tfjs'

export function L2regularization(w: Tensor, reg: number) {
  return w.mul(w).sum().mul(scalar(reg))
}
export function L1regularization(w: Tensor, reg: number) {
  return w.abs().sum().mul(scalar(reg))
}
export function L1L2regularization(w: Tensor, reg: number) {
  return L2regularization(w, reg).add(L1regularization(w, reg))
}

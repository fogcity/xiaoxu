export class Neuron{
 forward(inputs:any[]):
//   """ 假设输入和权重是1-D的numpy数组，偏差是一个数字 """
  const cell_body_sum = np.sum(inputs * self.weights) + self.bias
  const firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) 
  return firing_rate
}

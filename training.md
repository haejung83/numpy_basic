### 신경망 학습 (Neural-Net Training)

> 학습이란? 훈련 데이터로 부터 신경망의 가중치(weight)를 값을 최적화 하는 것을 말합니다.
>
> 여기서 신경망 학습의 중요한 지표인 손실함수(Loss Function)을 이해해야 합니다.

> 목표: 신경망의 학습 방법, 즉 데이터에서 매개변수를 경정하는 방법을 알아보자



* 기계학습은 데이터 주도 학습이다. 사람의 생각하는 알고리즘등에서 벗어난 데이터를 중심으로 접근하는 방법입니다. 이것의 의미하는것은 어떤 예측 기능을 구현함에 있어 사람이 설계한 알고리즘과 사람의 생각과 추론으로 만든 방법으로 그 기능을 수행하는 것이 아니라 기계가 해당하는 데이터를 기반하여 스스로 최적의 값을 찾는것으로 사람의 생각부분을 제거 한다는데 있습니다. 
* 정리하면 기계학습은 사람의 개입을 최소화하고 수집한 데이터로부터 패턴을 찾으려고 하는 것입니다. 



1. **데이터**
   * 데이터는 2종류로 구별하여 사용합니다.
     * 학습 데이터 (Training Data)
     * 시험 데이터 (Test Data)
   * 구분하는 이유
     * 범용 능력을 제대로 평가하기위해서
     * 학습한 모델이 훈련용 데이터에 과도하게 맞춰질 경우 실제 적요에서는 엉망의 결과를 나타낼 수 있습니다. 이런경우를 오버피팅(Overfitting)이라고 합니다.

2. **[손실 함수 (Loss Function)](loss_function.md)**

3. **[미니 배치 학습 (Mini-Batch)](mini_batch.md)**

4. **[수치 미분 (Numericla Differential)](numerical_differential.md)**

5. **[기울기 (Gradient)](gradient.md)**

6. 간단한 2층 신경망 구현하기

   * 신경망의 학습 단계 설명

     1. 미니배치
     2. 기울기 산출
     3. 가중치 매개변수 갱신
     4. 반복

   * 신경망의 학습이 이루어지는 순서입니다.

   * 경사 하강법으로 매개변수를 갱신하는 방법이며 이때 데이터를 미니배치로 무작위 선정하기 때문에
     확률적 경사 하강법 (Stochastic Gradient Descent)라고 부릅니다. 줄여서 SGD

   * code

     ```python
     import sys
     import os
     import numpy as np
     
     from activation import sigmoid
     from identity import softmax
     from loss import cross_entropy_error
     from gradient import numerical_gradient_batch as ng
     
     # Referenced from Deep learning from scratch
     
     class TwoLayerNet:
     
         def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
             self._params = dict()
             # First layer
             self._params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
             self._params['b1'] = np.zeros(hidden_size)
     
             # Second layer
             self._params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
             self._params['b2'] = np.zeros(output_size)
     
         def predict(self, x):
             W1, W2 = self._params['W1'], self._params['W2']
             b1, b2 = self._params['b1'], self._params['b2']
     
             a1 = np.dot(x, W1) + b1
             z1 = sigmoid(a1)
             a2 = np.dot(z1, W2) + b2
             z2 = sigmoid(a2)
     
             y = softmax(z2)
             return y
     
         def loss(self, x, t):
             y = self.predict(x)
             return cross_entropy_error(y, t)
     
         def accuracy(self, x, t):
             y = self.predict(x)
             y = np.argmax(y, axis=1)
             t = np.argmax(t, axis=1)
     
             accuracy = np.sum(y == t) / float(x.shape[0])
             return accuracy
     
         def numerical_gradient(self, x, t):
             loss_W = lambda W: self.loss(x, t)
     
             grads = dict()
             grads['W1'] = ng(loss_W, self._params['W1'])
             grads['b1'] = ng(loss_W, self._params['b1'])
             grads['W2'] = ng(loss_W, self._params['W2'])
             grads['b2'] = ng(loss_W, self._params['b2'])
     
             return grads
     
     ```

   * Test

     ```python
     import numpy as np
     
     net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
     
     x = np.random.rand(100, 784) # 입력 데이터 
     t = np.random.rand(100, 10)  # 정답
     
     grads = net.numerical_gradient(x, t) # 기울기 계산 (오래 걸림)
     ```

   * 중앙 차분형태의 수치 미분 방식은 성능이 너무 않좋습니다.

   * 이후에 오차역전파(Back Propagation)을 이용한 고속 기울기 방법을 알아 보겠습니다.

7. 학습의 단위 Epoch

   * Epoch(에폭)은 하나의 단위로서 1 에폭은 학습 데이터를 전부 소진했을 때의 횟수를 이야기 합니다.
   * 예로 10000개의 학습 데이터가 있고 미니배치 사이즈는 100일 경우 미니배치 학습을 100번 수행하면 100데이터 * 100번 = 10000 즉 모든 학습 데이터가 소진된 이 시점을 1 에폭이라고 합니다.







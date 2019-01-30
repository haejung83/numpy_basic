### 최적화 기법

최적화란?

> 신경망 학습의 목적은 손실 함수의 값을 가능한 한 낮추는 매개변수를 찾는 것입니다. 이는 곧 매개변수의 최적값을 찾는 문제이며, 이러한 문제를 푸는 것을 최적화(**Optimization**)라 합니다.
>
> 지금까지 최적의 매개변수 값을 찾는 단서로 매개변수의 기울기(미분)을 이용했습니다. 매개변수의 기울기를 구해 기울어진 방향으로 매개변수 값을 갱신하는 일을 몇 번이고 반복해서 점점 최적의 값에 다가 갔습니다. 
>
> 이 방법을 확률적 경사 하강법(SGD)입니다. 현재 SGD이외의 여러 최적화 방법이 존재하며 간단히 알아보도록 하겠습니다.



1. 확률적 경사 하강법 (Stochastic Gradient Descent, SGD)

   * 식
     $$
     W{\leftarrow}W-\eta\frac{\partial L}{\partial W}
     $$

   * W는 갱신할 매개변수, eta는 학습률(Learning Rate) 그리고 W에 대한 손실함수의 **기울기**

   * eta, 즉 학습률은 0.01, 0.001등으로 미리 고정하여 사용합니다.

   * code

     ```python
     class SGD:
         def __init__(self, lr=0.01):
             self._lr = lr
             
         def update(self, params, grads):
             for key in params.keys():
                 params[key] -= self.lr * grads[key]
     ```

   * 단점

     * 문제에 따라 비효율적일 때가 있다. 기울기가 Y축으로만 심하고 X축으로 거의 없을 경우 엄청난 지그재그 형태를 보입니다.

2. 모멘텀 (Momentum)

   * 모멘텀의 뜻은 운동량을 나타냅니다. 즉, 속도의 개념을 도입한 것 입니다.

   * 식
     $$
     v{\leftarrow}\alpha v-\eta\frac{\partial L}{\partial W} \\
     W{\leftarrow}W + v
     $$

   * v는 속도

   * alpha*v는 물체가 아무런 힘을 받지 않을때 서서히 하강시키는 역활을 합니다.

   * code

     ```python
     class Momentum:
         def __init__(self, lr=0.01, momentum=0.9):
         	self.lr = lr
             self.momentum = momentum
             self.v = None
             
         def update(self, params, grads):
             if self.v is None:
                 self.v = dict()
                 for key, val in params.items():
                     self.v[key] = np.zeros_like(val)
                     
                 for key in params.keys():
                     self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
                     params[key] += self.v[key]
     ```

     

3. AdaGrad (Adaptive Gradient)

   * 각각의 매개변수에 맞춤형으로 매개변수의 학습률을 조정하는 방법입니다.

   * 식
     $$
     h{\leftarrow}h+\frac{\partial L}{\partial W} \cdot \frac{\partial L}{\partial W} \\
     W{\leftarrow}W-\eta\frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}
     $$

   * h는 기울기 값을 계속 제곱하여 더해줍니다.

   * 그리고 매개변수를 갱신할 때 1/sqrt(h)를 곱하여 학습률을 조정합니다. 즉 매개변수의 원소중 많이 움직인 원소는 학습률이 낮아진다는 뜻입니다.

   * code

     ```python
     class AdaGrad:
         def __init__(self, lr=0.01):
             self.lr = lr
             self.h = None
             
         def update(self, params, grads):
             if self.h is None:
                 self.h = {}
                 for key, val in parmas.items():
                     self.h[key] = np.zeros_like(val)
                     
             for key in params.keys():
                 self.h[key] += grads[key] * grads[key]
                 params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))
     ```

4. RMSProp

   * AdaGrad는 과거의 기울기를 제곱하여 계속 더합니다. 그래서 학습을 진행할 수록 갱신 강도가 약해집니다. 실제로 무한히 계속 학습한다면 어느순간 갱신량이 0이 되어 전혀 갱신되지 않게 되죠. 

   * 이문재를 개선한 기법으로 RMSProp이 있습니다. 과거의 모든 기울기를 균일하게 더해가는 것이 아니라, 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영합니다.

   * 이를 지수이동평균(Exponential Moving Average)이라 하여 과거의 기울기의 반영 규모를 기하급수적으로 감소시킵니다.

   * code

     ```python
     class RMSprop:
         def __init__(self, lr=0.01, decay_rate = 0.99):
             self.lr = lr
             self.decay_rate = decay_rate
             self.h = None
             
         def update(self, params, grads):
             if self.h is None:
                 self.h = {}
                 for key, val in params.items():
                     self.h[key] = np.zeros_like(val)
                 
             for key in params.keys():
                 self.h[key] *= self.decay_rate
                 self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
                 params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
     ```

5. Adam (Adaptive Momentum)

   * 모먼텀은 운동량, 속도의 개념을, AdaGrad는 매개변수의 개신 정도(학습률)을 조정했습니다. 그럼 이 두개를 융합하면 어떨까요? 그것이 Adam입니다.

   * code

     ```python
     class Adam:
         def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
             self.lr = lr
             self.beta1 = beta1
             self.beta2 = beta2
             self.iter = 0
             self.m = None
             self.v = None
             
         def update(self, params, grads):
             if self.m is None:
                 self.m, self.v = {}, {}
                 for key, val in params.items():
                     self.m[key] = np.zeros_like(val)
                     self.v[key] = np.zeros_like(val)
             
             self.iter += 1
             lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
             
             for key in params.keys():
                 self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
                 self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
                 params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
     ```



#### 어느 갱신 방법을 이용하는게 좋을까요?

다순히 결과만 보면 AdaGrad가 가장 좋아보입니다. 하지만 풀어야할 문제가 어떤가에 따라 달라집니다. 현재 모든 문제에 대하여 항상 뛰어난 결과를 주는 기법은 없습니다. 각자의 장단이 있어 잘푸는 문제와 서툰 문제가 있습니다.

요즘에는 많은 사람들이 Adam에 만족해하며 쓰는 것 같습니다. 

하지만 명심하세요. 사용하는 학습률, 그리고 신경망의 구조(깊이)에 따라 최적화 방법은 달라집니다.
---
typora-root-url: plot_images
---

### 활성화 함수 (Activation Function)

> 입력 신호의 총 합을 특정 임계점에 대응하여 신호의 활성화를 결정한다. 즉, 들어오는 입력신호의(1~n개) 총합이 특정 임계점 이상을 넘어서면 1, 그렇지 않으면 0을 출력하게 한다.

* 종류
  * Sigmoid
  * Step
  * ReLU (Rectified Linear Unit)
  * 등등(Elu, Relu6, Etc....)



1. Sigmoid

   * 식
     $$
     \mathbf{h}(x) = \frac{1}{1 + e^{-x}}
     $$

   * exp는 자연상수 exponential을 의미 (2.718281828)

     ![sigmoid plt](sigmoid_plt.png)

   * code

     ```python
     import numpy as np
     
     def sigmoid(x):
         return 1 / (1+np.exp(-x))
     ```

2. Step

   * x가 0보다 크면 무조건 1, 0이거나 0보다 작으면 0을 출력
     ![step plt](step_plt.png)

   * code

     ```python
     import numpy as np
     
     def step(x):
         y = x > 0
         return y.astype(np.int)
     ```

3. ReLU (Rectified Linear Unit)

   * 식
     $$
     h(x)= 
     \begin{cases}
         x  & (x \gt 0)\\
         0  & (x \le 0)
     \end{cases}
     $$

   * x가 0보다 크면 x를 그대로 출력하고, x가 0과 같거나 작으면 0을 출력한다.
     ![relu plot](relu_plt.png)

   * code

     ```python
     import numpy as np
     
     def relu(x):
         return np.maximum(0, x)
     ```

### 항등함수 (Identity Function)

> 최종 출력층에 사용하는 함수
>
> 기계학습 문제에 대한 목적이 분류 / 회귀에 따라 구분할 수 있다.
>
> 일반적으로 분류에는 Softmax 함수를, 회귀에는 Identity 함수를 사용한다.



* 종류
  * Identity (회귀)
  * Sigmoid (2진 분류)
  * Softmax (다중 분류)



1. Identity

   * 식
     $$
     y = x
     $$

   * code

     ```python
     def identity(x)
         return x
     ```

2. Sigmoid

   * 생략

3. Softmax

   * 식
     $$
     y_k = \frac{exp(a_k)}{\sum_{i=1}^{n}{exp(a_i)}}
     $$

   * code

     ```python
     import numpy as np
     
     def softmax(x):
         xmax = np.max(x) # for preventing overflow
         xnexp = np.exp(x-xmax)
         return xnexp / np.sum(xnexp)
     ```



### 수치 미분

1. 미분

   - 시간공간속에서 어느 순간의 변화량(기울기)를 구하는 것이다.

   - 식
     $$
     \frac{df(x)}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
     $$

   - 해석적(analytic) 미분

     - 진정한 미분 (우리가 교과서에서 배우던..)
     - 어느 순간! 그 순간의 기울기
     - x^2은 2x와 같이 해석적인 풀이로 얻는 것

   - 수치(numerical) 미분

     - 컴퓨터로 가능한한 작은 단위의 두 지점의 기울기(차)로 구한다.

     - 전방 차분

       - 예)
         $$
         \lim_{h\to0}\frac{f(x+h)-f(x)}{h}
         $$

       - code

         ```python
         import numpy as np
         
         def numerical_diff(f, x):
             h = 1e-4 # 0.0001
             return (f(x+h)-f(x))/h
         ```

     - 중앙 차분

       - 예)
         $$
         \lim_{h\to0}\frac{f(x+h)-f(x-h)}{2h}
         $$

       - code

         ```python
         import numpy as np
         
         def numerical_diff(f, x):
             h = 1e-4 # 0.0001
             return (f(x+h)-f(x-h))/(2*h)
         ```

     - 오차를 가급적 줄이기 위해 중앙 차분을 이용한다. 

     - 컴퓨터에서는 부동소수점 반올림 오차로 인해서 무한정 작은 단위의 h를 사용할 수 없다. 

       ```python
       >>> np.float32(1e-0) #반올림 오차가 발생한다. 
       0.0
       ```

     - 대체로 h를 0.0001로 하면 좋은 결과를 얻는다고 한다.

     - 수치미분

       - 예

       $$
       y=0.01x^2+0.1x
       $$

       - 그래프 x가 10일때
         ![numerical_diff_10_plt](./plot_images/numerical_diff_10_plt.png)

2. 편미분

   - 함수의 변수가 2개 이상일 경우 각각의 변수를 개별로 하여 나머지 변수는 상수로 하고 기울기를 구한다.

   - 예)
     $$
     f(x_0, x_1)=x_0^2+x_1^2
     $$

     ```python
     def multival_function(x):
         return x[0]**2 + x[1]**2
     ```

   - 위와 같은 함수가 있고 여기서 x0는 3, x1은 4일 때, **x0**에 대한 편미분을 구하라

     ```python
     >>> def multival_function(x0):
            return x0**2 + 4**2 # x1인 4는 상수가 된다. 
     
     >>> numerical_diff(multival_function, 3.0)
     6.00000000000...
     ```

   - 위와 같은 함수가 있고 여기서 x0는 3, x1은 4일 때, **x1**에 대한 편미분을 구하라

     ```python
     >>> def multival_function(x1):
            return 3**2 + x1**2 # x0인 3은 상수가 된다. 
     
     >>> numerical_diff(multival_function, 4.0)
     7.99999999999...
     ```

   - 이처럼 다변수일 때 구하고자 하는 변수를 제외한 다른 변수를 상수로 두고 구하고자 하는 변수에 대한 기울기를 구하는것이 편미분 이다.
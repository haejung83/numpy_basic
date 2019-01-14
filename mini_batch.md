### 미니 배치 학습 (Mini-Batch)

- 배치(Batch)는 묶음 단위다. 

- 학습에서 다수개의 훈련 이미지에 대하여 1개식 손실 함수를 구하고 그 값을 최대한 줄여주는 매개변수를 찾게 된다. 이것을 학습 이미지가 10000개라면 10000번 하게 된다.

- 이렇게 일일이 손실함수를 구하는것은 현실적이지 않기에 데이터 일부를 추려서 근사치로 이용하는 방법을 사용한다. 

- 이렇게 훈련 데이터의 일부만 골라 학습을 수행할 때 이 일부를 미니배치(Mini-Batch)라 한다.

- 미니 배치용 교체 엔트로피

  - 식
    $$
    E = -\frac{1}{N}\sum_{n}\sum_{k}t_{nk}\log{y_{nk}}
    $$

  - code

  - one-hot encoding 으로 입력 받는 경우 ex) [0, 0, 0, 1]

    ```python
    import numpy as np
    
    def cross_entropy_error_by_onehot(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
    
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y)) / batch_size
    ```

    - 정답과 결과가 1차원 배열인경우 2차원 배열로 만든다.
    - 각각의 CEE를 구하고 batch_size로 평균을 구한다.

  - 숫자 레이블로 입력 받는 경우 ex) 3

    ```python
    import numpy as np
    
    def cross_entropy_error_by_index(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
    
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_szie), t])) / batch_size
    ```

  - 마찬가지로 입력된 정답과 결과과 1차원 배열인경우 2차원 배열로 만든다. 

  - y[np.arange(batch_size), t]는 np.arange로 인덱스를 만들고 각 인덱스별 t의 위치에 해당하는 값만 가지고 와서 log를 구한다. 나머지 부분은 무시
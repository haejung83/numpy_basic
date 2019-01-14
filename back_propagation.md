#### 역전파 (Back Propagation)

> 가중치 매개변수의 기울기를 효율적으로 계산하는 방법입니다. 먼저 구현했던 수치미분은 구현하기 쉬운반면 속도면에서 많이 느립니다. 이를 개선하기 위해서 오차 역전파 방법을 사용합니다.



1. 계산 그래프
   1. 정의
   2. 국소적 계산
2. 연쇄법칙 (Chain Rule)
3. 역전파 (Back Propagation)
   1. 덧셈 노드
   2. 곱셈 노드
4. 코드로 구현하기
   1. 덧셈 계층
   2. 곱셈 계층
   3. 활성화 함수 계층
      1. ReLU
      2. Sigmoid
   4. Affine/Softmax 계층
      1. Affine
      2. Softmax
   5. Batch Affine 계층
   6. Softmax with Loss 계층
5. 최종 구현
6. 테스트
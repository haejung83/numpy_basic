import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    h2 = 2.0*h
    grad = np.zeros_like(x) #x랑 같은 크기의 배열을 생성하고 0으로 초기화
    
    for idx in range(x.size):
        tmp_val = x[idx] 
        x[idx] = tmp_val + h # f(x+h)
        fxh1 = f(x) # x를 제외한 나머지는 상수로 두고 기울기를 계산
        
        x[idx] = tmp_val - h # f(x-h)
        fxh2 = f(x) # 마찬가지
        
        grad[idx] = (fxh1 - fxh2) / h2 # 중앙차분으로 기울기를 구함
        x[idx] = tmp_val # 복원
        
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x
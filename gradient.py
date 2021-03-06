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

def numerical_gradient_batch(f, X):
    if X.ndim == 1:
        return numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        print('X.shape: {}'.format(X.shape))
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient(f, x)
        return grad


def numerical_gradient_test(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x
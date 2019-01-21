import numpy as np 
from layer_naive import MulLayer, AddLayer

# Simple problem (apple price with tax)
'''
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price_with_tax = mul_tax_layer.forward(apple_price, tax)

print('MulLayer - price with tax: {}'.format(price_with_tax))

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print('MulLayer - dapple: {}, dapple_num: {}, dtax: {}'.format(dapple, dapple_num, dtax))
'''

# More complex problem
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Forward
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
apple_orange_with_tax = mul_tax_layer.forward(apple_orange_price, tax)

print('Apple Orange - price with tax: {}'.format(apple_orange_with_tax))

dprice = 1

dapple_orange_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dapple_orange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)


print('Apple Orange - dapple: {}, dapple_num:{}, dorange: {}, dorange_num:{}, dtax:{}'
    .format(dapple, dapple_num, dorange, dorange_num, dtax))

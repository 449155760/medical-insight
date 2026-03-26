# PPOloss

# GAE
#self Attension
#attention(Q,K,V)=softmax(QK^T/squat(d_k))V

# 梯度下降 根号x g(x)=y^2-x
def func(x):
    learning_rate=0.01
    y=x
    max_iterations=100
    for _ in range(max_iterations):
        gradient=2*y
        new_y=y-learning_rate*gradient
        if abs(new_y-y)<1e-4:
            break
        y=new_y
    return y
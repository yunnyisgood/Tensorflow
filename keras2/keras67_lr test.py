weight = 0.7 # -> 값이 커질수록 빠르게 goal_pred 에 도달
input = 0.5 # 초기값
goal_pred = 0.8 # pred 값 이외에 모두 튜닝 가능
lr = 0.001 
# 학습률 -> 값이 0.2, 0.25이렇게 커지게 되면 목표값인 0.8을 지나치게 될 수도!
#        -> 0.1보다 0.001을 했을 때 pred 증가폭이 훨씬 작기 때문에 목표치에 도달하기 위해서는 epochs를 훨씬 많이 늘려줘야 한다 
epochs = 100

for i in range(epochs):
    pred = input * weight
    error = (pred - goal_pred)**2

    print(str(i)+"\tError:" + str(error) + "\tPrediction: " + str(pred))

    up_pred = input * (weight + lr) # 햑습률을 더한상태
    up_error = (goal_pred - up_pred) ** 2

    down_pred = input * (weight - lr) # 학습률을 뺀상태
    down_error = (goal_pred - down_pred) ** 2

    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr

'''

0       Error:0.20250000000000007       Prediction: 0.35
1       Error:0.19802500000000006       Prediction: 0.355
2       Error:0.19360000000000005       Prediction: 0.36
3       Error:0.18922500000000006       Prediction: 0.365
4       Error:0.18490000000000004       Prediction: 0.37
5       Error:0.18062500000000004       Prediction: 0.375
6       Error:0.17640000000000003       Prediction: 0.38
7       Error:0.17222500000000002       Prediction: 0.385
8       Error:0.16810000000000003       Prediction: 0.39
9       Error:0.16402500000000003       Prediction: 0.395

-> 92번째에 prediction 도달

'''        





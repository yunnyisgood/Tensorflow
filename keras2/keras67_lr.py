weight = 0.5 # 가중치
input = 0.5 # 초기값
goal_pred = 0.8 # pred 값 이외에 모두 튜닝 가능
lr = 0.01 # 학습률
epochs = 10

for i in range(epochs):
    pred = input - weight
    error = (pred - goal_pred)**2

    print("Error:" + str(error) + "\tPrediction: " + str(pred))

    up_pred = input * (weight + lr) # 햑습률을 더한상태
    up_error = (goal_pred - up_pred) ** 2

    down_pred = input * (weight - lr) # 학습률을 뺀상태
    down_error = (goal_pred - down_pred) ** 2

    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight + lr

'''
Error:0.6400000000000001        Prediction: 0.0
Error:0.6561000000000001        Prediction: -0.010000000000000009
Error:0.6724000000000001        Prediction: -0.020000000000000018
Error:0.6889000000000001        Prediction: -0.030000000000000027
Error:0.7056000000000001        Prediction: -0.040000000000000036
Error:0.7225000000000001        Prediction: -0.050000000000000044
Error:0.7396000000000001        Prediction: -0.06000000000000005
Error:0.7569000000000002        Prediction: -0.07000000000000006
Error:0.7744000000000002        Prediction: -0.08000000000000007
Error:0.7921000000000002        Prediction: -0.09000000000000008

weight = 0.1 # 가중치
input = 0.5 # 초기값
Error:0.16000000000000003       Prediction: 0.4
Error:0.16810000000000003       Prediction: 0.39
Error:0.17640000000000003       Prediction: 0.38
Error:0.18490000000000004       Prediction: 0.37
Error:0.19360000000000005       Prediction: 0.36
Error:0.20250000000000007       Prediction: 0.35
Error:0.21160000000000007       Prediction: 0.33999999999999997
Error:0.22090000000000007       Prediction: 0.32999999999999996
Error:0.23040000000000008       Prediction: 0.31999999999999995
Error:0.2401000000000001        Prediction: 0.30999999999999994

'''        





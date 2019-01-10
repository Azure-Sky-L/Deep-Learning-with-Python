k = 4
num_validation_samples = len(data) // k

np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold:  #选择验证数据分区
        num_validation_samples * (fold + 1)]
    training_data = data[:num_validation_samples * fold] +
        data[num_validation_samples * (fold + 1):]   # 使用剩余数据作为训练数据。注意, + 运算符是列表合并,不是求和
    model = get_model() #创建一个全新的模型
    model.train(training_data)
    validation_score = model.evaluate(validation_data) 
    validation_scores.append(validation_score) # 实例(未训练)
validation_score = np.average(validation_scores) # 最终验证分数:K 折验证 分数的平均值
model = get_model() # 在所有非测试数据上训练最终模型
model.train(data)
test_score = model.evaluate(test_data)

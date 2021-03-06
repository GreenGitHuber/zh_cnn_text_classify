import pandas as pd
from sklearn.ensemble import IsolationForest
ilf = IsolationForest(n_estimators=100,
                      n_jobs=-1,          # 使用全部cpu
                      verbose=2,
    )
data = pd.read_csv('feature.csv', index_col="id")
data = data.fillna(0)
# 选取特征，不使用标签(类型)
# X_cols = ["age", "salary", "sex"]
print(data.shape)

# 训练
# ilf.fit(data[X_cols])
# ilf.fit(data)

# shape = data.shape
# batch = 10**2

# all_pred = []
# for i in range(shape/batch+1):
#     start = i * batch
#     end = (i+1) * batch
#     test = data[start:end]
#     # 预测
#     pred = ilf.predict(test)
#     all_pred.extend(pred)

# data['pred'] = all_pred
# data.to_csv('outliers.csv', columns=["pred",], header=False)


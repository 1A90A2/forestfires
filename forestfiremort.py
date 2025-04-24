# 1-1 데이터 불러오기 및 로그 변환
import pandas as pd
import numpy as np

fires = pd.read_csv("./sanbul2district-divby100.csv")
fires['burned_area'] = np.log(fires['burned_area'] + 1)
fires["month"] = fires["month"].str.extract(r"(\d+)", expand=False).astype(int)
fires["day"] = fires["day"].str.extract(r"(\d+)", expand=False).astype(int)

# 1-2 탐색
print(fires.head())
print(fires.info())
print(fires.describe())

print(fires['month'].value_counts())
print(fires['day'].value_counts())


# 1-3 시각화
import matplotlib.pyplot as plt
import seaborn as sns

fires.hist(bins=50, figsize=(15,10))
plt.suptitle("Histogram of Features")
plt.show()

sns.scatterplot(x='avg_temp', y='max_wind_speed', data=fires, hue='burned_area')
plt.title("Avg Temp vs Max Wind colored by Burned Area")
plt.show()

# 1-4 burned_area 로그 변환
fires['log_area'] = np.log1p(fires['burned_area'])

# 비교 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(fires['burned_area'], bins=50, kde=True)
plt.title("Original 'burned_area'")

plt.subplot(1,2,2)
sns.histplot(fires['log_area'], bins=50, kde=True)
plt.title("Log-Transformed 'log_area'")

plt.tight_layout()
plt.show()

# 1-5 train_test_split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
test_set.head()
fires["month"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
 strat_train_set = fires.loc[train_index]
 strat_test_set = fires.loc[test_index]
print("\nMonth category proportion: \n",
 strat_test_set["month"].value_counts()/len(strat_test_set))
print("\nOverall month category proportion: \n",
 fires["month"].value_counts()/len(fires))

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# 1-6 수치형 특성 중에서 일부 선택 + log_area 포함
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


features = ['avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind', 'log_area']
scatter_matrix(fires[features], figsize=(12, 8), diagonal='kde')
plt.suptitle("Scatter Matrix of Selected Features", fontsize=14)
plt.show()

# 1-7 지도 시각화 (원의 크기: max_temp, 색상: burned_area)
plt.figure(figsize=(10, 8))
plt.scatter(
    fires['longitude'], fires['latitude'], 
    s=fires['max_temp'] * 10,              # 원 크기 = max_temp  = s
    c=fires['log_area'],                   # 색상 = log_area  = c
    cmap='YlOrRd', alpha=0.4
)
plt.colorbar(label='Log Burned Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Wildfires by Location (Size: Max Temp, Color: Log Burned Area)')
plt.grid(True)
plt.show()

# 1-8 OneHotEncoder 적용 (month, day)
from sklearn.preprocessing import OneHotEncoder

fires = strat_train_set.drop(["burned_area"], axis=1)  # 또는 "log_area"
fires_labels = strat_train_set["burned_area"].copy()   # 또는 "log_area"

fires_num = fires.drop(["month", "day"], axis=1)
fires_cat = fires[["month", "day"]]

cat_encoder = OneHotEncoder(sparse=False)
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)

OneHotEncoder(handle_unknown='ignore', dtype=np.float64)

print("Encoded categorical feature shape:", fires_cat_1hot.shape)
print("Encoded categories:", cat_encoder.get_feature_names_out(["month", "day"]))

# 1-9 Pipeline (수치 + 범주형 전처리)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 수치형 전처리 pipeline
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# 수치형 / 범주형 특성 나누기
num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']


# 전체 전처리 파이프라인 구성 (수치형 + 범주형)
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# 이후 OneHotEncoder 설정
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float64)

import joblib
full_pipeline.fit(fires)
joblib.dump(full_pipeline, 'full_pipeline.pkl')


# 파이프라인 적용
fires_prepared = full_pipeline.fit_transform(fires)

print("\n\n########################################################################")
print("fires_prepared shape:", fires_prepared.shape)


# Keras model 개발
import tensorflow as tf
from tensorflow import keras

# 테스트 세트 준비
fires_test = strat_test_set.drop(["burned_area"], axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()

# 전처리 파이프라인 적용
fires_test_prepared = full_pipeline.transform(fires_test)


X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

# Keras 모델 저장
model.save('fires_model.keras')

# evaluate model
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n", 
      np.round(model.predict(X_new), 2))

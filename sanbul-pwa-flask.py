# sanbul-pwa-flask.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms import FloatField, IntegerField


# 모델 및 전처리기 로드 (한 번만)
model = keras.models.load_model('fires_model.keras')
pipeline = joblib.load('full_pipeline.pkl')  # 저장된 ColumnTransformer

# Flask 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap5(app)

# 폼 클래스 정의
class LabForm(FlaskForm):
    longitude = StringField('longitude (1-7)', validators=[DataRequired()])
    latitude = StringField('latitude (1-7)', validators=[DataRequired()])
    month = StringField('month (1-12)', validators=[DataRequired()])
    day = StringField('day (0-6, 0:sun, 1:mon, ...)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

# 메인 페이지
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# 예측 페이지
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    res = None

    if form.validate_on_submit():
        # 입력 값 수집
        data = {
            'longitude': float(form.longitude.data),
            'latitude': float(form.latitude.data),
            'month': int(form.month.data),
            'day': int(form.day.data),
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data),
        }

        # 데이터프레임 생성
        input_df = pd.DataFrame([data])
        print(input_df)
        print(input_df.dtypes)
        print(input_df.isna())


        # 전처리 적용
        input_prepared = pipeline.transform(input_df)

        # 예측
        prediction = model.predict(input_prepared)
        real_area = np.expm1(prediction[0][0])  # 로그 역변환
        res = round(real_area, 2)

        return render_template('result.html', form=form, res=res)

    return render_template('prediction.html', form=form)

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True)

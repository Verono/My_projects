#Загрузка необходимых библиотек
import os

from typing import List
from fastapi import FastAPI
from datetime import datetime

from catboost import CatBoostClassifier

from pydantic import BaseModel

import pandas as pd

import psycopg2

from loguru import logger
from sqlalchemy import create_engine

import hashlib



app = FastAPI()

#Функция для работы с директориями
def get_model_path_control(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = '/workdir/user_input/model_control'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def get_model_path_test(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/model_test'
    else:
        MODEL_PATH = path
    return MODEL_PATH


class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    
    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


#Загрузка подготовленных фичей для контрольной и тестовой модели
def load_features_test(): 
    query_1 = "SELECT * FROM public.meta_ve" 
    features= pd.read_sql(query_1,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml",
        )
    return features


def load_features_control(): 
    query_1 = "SELECT * FROM public.meta_control" 
    features= pd.read_sql(query_1,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml",
        )
    return features

'''pd.read_sql неоптимально (бесконечное увеличение из-за большого числа копирований),
поэтому таблицы занимают в 5-6 раз больше места, чем нужно.
Чтобы этого избежать, создадим функцию, которая считывает кусочками'''

#Функция для загрузки таблиц частями
def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)

#Функция для загрузки таблицы Users
def load_users():
    df_user = pd.read_sql(
    """SELECT * FROM public.user_data""",
    con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
        )
    return df_user

#Функция для загрузки таблицы Feed
def load_feed():
    logger.info("load feed_data")
    query_4 = """SELECT * FROM public.feed_data where action='like' limit 1000000;"""
    features = batch_load_sql(query_4)
    return features

#Функция для загрузки тестовой модели
def load_models_test():
    model_path = get_model_path_test("D:/StartML_KarpovCourses/MachineLearning/Урок_22_рекомендательные_системы/My_Project/catboost_model_with_16_clasters_3_hour_month.cbm")
    #model_path = get_model_path_test("/my/super/path")
    # LOAD MODEL HERE PLS :)
    from_file =  CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    from_file.load_model(model_path)
    return from_file

#Функция для загрузки контрольной модели
def load_models_control():
    model_path = get_model_path_control("D:/StartML_KarpovCourses/MachineLearning/Урок_22_рекомендательные_системы/My_Project/catboost_model_with_TDF_hour_month.cbm")
    #model_path = get_model_path_control("/my/super/path")
    # LOAD MODEL HERE PLS :)
    from_file =  CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    from_file.load_model(model_path)
    return from_file

#Загрузка моделей
model_control = load_models_control()
print(model_control)
model_test = load_models_test()
print(model_test)

#Загрузка feed
liked_posts_all=load_feed()
print(liked_posts_all)

#Загрузка постов + фичи
posts_features_all_test=load_features_test()
posts_features_all_control=load_features_control()
print(posts_features_all_test)
print(posts_features_all_control)

#Загрузка users
user_features_all=load_users()
print(user_features_all)
  
def get_recommended_posts(id: int, time: datetime, limit: int, exp_group: str):
    if exp_group == 'control':
        posts_features_all=posts_features_all_control
        model=model_control
    else:
        posts_features_all=posts_features_all_test
        model=model_test
    
    #Загрузим фичи по пользователям
    user_features=user_features_all.loc[user_features_all.user_id==id]
    user_features=user_features.drop('user_id', axis=1)
    
    #Загрузим фичи по постам
    posts_features=posts_features_all.drop(['index','text'],axis=1)
    content=posts_features_all[['post_id','text','topic']]
    
    #Объединим эти фичи
    add_user_features=dict(zip(user_features.columns,user_features.values[0]))
    user_posts_features=posts_features.assign(**add_user_features)
    user_posts_features=user_posts_features.set_index(['post_id'])
    
    #Добавим еще фичи
    user_posts_features['hour']=time.hour
    user_posts_features['month']=time.month
    
    
    #Сформируем предсказания вероятности лайкнуть пост для всех постов
    predicts=model.predict_proba(user_posts_features)[:,1]
    user_posts_features['predicts']=predicts
    
    #Уберем записи, где пользователь уже ставил лайк
    liked_posts=liked_posts_all
    filtered=user_posts_features[~user_posts_features.index.isin(liked_posts)]
    
    #Рекомендуем топ по вероятности постов
    recommended_posts=filtered.sort_values('predicts')[-limit:].index
    
    return Response(
        recommendations=[
            PostGet(
                id=i, 
                text=content[content.post_id==i].text.values[0], 
                topic=content[content.post_id==i].topic.values[0]
            )
            for i in recommended_posts
        ],
        exp_group=exp_group
        )
        


#функция, которая по user_id пользователя определяет, в какую группу попал пользователь
salt='my_salt'

def get_exp_group(user_id: int) -> str:
    value_str = str(user_id) + salt
    percent = int(hashlib.md5(value_str.encode()).hexdigest(), 16) % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    exp_group = get_exp_group(id)
    if exp_group == 'control':
      recommendations = get_recommended_posts(id, time, limit, exp_group)
    elif exp_group == 'test':
      recommendations = get_recommended_posts(id, time, limit, exp_group)
      print(to_mb(recommendations))
    else:
      raise ValueError('unknown group')
    return recommendations

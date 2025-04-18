import streamlit as st
import pickle
import numpy as np

# import model
df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))


st.title('Laptop Price Predictor')

# Company
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
types= st.selectbox('Type',df['TypeName'].unique())

# Ram
ram=st.selectbox('Ram in GB',[2,4,6,8,12,16,24,32,64])

#Weight
weight=st.number_input('Weight of the Laptop (kg)')

#TouchScreen
touchScreen=st.selectbox('Touch-Screen',['No','Yes'])

# ips
ips=st.selectbox('Ips',['No','Yes'])

# screen_size
screen_size=st.number_input('Screen-Size (inches)')

# resolution
resolution=st.selectbox('Screen-Resolution',[
    '1024x600','1280x720','1366x768','1600x900','1920x1080','1920x1200',
    '2048x1080','2160x1440','2560x1440','2560x1600','2880x1620','2880x1800',
    '3200x1800','3240x2160','3840x2160','4096x2160','5120x2880'])

# Cpu_brand
cpu=st.selectbox('CPU',df['Cpu_brand'].unique())

# hdd
hdd=st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

# ssd
ssd=st.selectbox('SSD (in GB)',[0,8,128,256,512,1024,2048])

# GPU brand
gpu=st.selectbox('GPU',df['Gpu_brand'].unique())

#OS
os=st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    if 'touchScreen' == 'Yes':
        touchScreen=1
    else:
        touchScreen=0

    if 'ips' =='Yes':
        ips=1
    else:
        ips=0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi= ((X_res**2) + (Y_res**2)) **0.5/screen_size

    query=np.array([company,types,ram,weight,touchScreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)

    st.title(int(np.exp(pipe.predict(query)[0])))

import os

import cv2
import numpy as np
import streamlit as st
import torch
import zipfile

from model.model import MobileNetV2Model

print(os.getcwd())
print(os.listdir('.'))
cla_dict = {0: '猫', 1: '狗'}
print(cla_dict)
# st.write(str(os.listdir('weights')))
device = torch.device('cpu')
model = MobileNetV2Model(num_classes=2).to(device)
# try:
#     checkpoint = torch.hub.load_state_dict_from_url('https://public.ch.files.1drv.com/y4mapAGWfxALhSFloP0PDSg9HmM7PDe7SjiCZePZ6kYcCXW1rbMF50OUdFEXX_dUdTqdbBiaGOYWx_cgVK8jYzKkzX2fNurulS-LDJ2JlgH2nX0A7rW_p6WzlxKO5ThP9ewhH3z9ntU-FBmypiJEGyQK5hMR57q5XRgh2ogHmullFH5M129sVVmUXNm8qTo1SdGaaWjFYzMm1fCDssJReRDOATorTlTnnrPVI-6NTykWRo', map_location=device)
# except:
#     with zipfile.ZipFile('model_best.zip', 'r') as z:
#         ckpt = z.extract('model_best.pth')
#     checkpoint = torch.load(ckpt, map_location=device)
# state_dict = checkpoint['state_dict']
# model.load_state_dict(state_dict)

# prepare model for testing
model.eval()


def img_transform(img):
    img = cv2.resize(img, (224, 224))
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


def predict(img):
    img = img_transform(img)
    img = img.to(device)
    output = model(img)
    pred = torch.argmax(output, dim=1)
    return pred


def read_img(img_path):  # 读取中文路径图片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    return img


def main():
    st.title('黄超睿的猫狗分类Demo')
    st.number_input('图片标号', step=1)
    '请点击页面左上角展开侧边栏以加载数据'
    with st.sidebar:
        env = st.selectbox('选择环境', ['服务器运行','本地运行'])
        if env == '服务器运行':
            st.header('选择数据')
            file_bytes = st.file_uploader('上传图片', type=['jpg', 'png', 'jpeg'])
        elif env == '本地运行':
            try:
                st.header('选择数据')
                file_bytes = st.file_uploader('上传图片', type=['jpg', 'png', 'jpeg'])
                test_path = st.text_input('输入测试集路径', value=r'E:\github\pytorch深度学习实验内容\datasets\test')
                num = int(st.number_input('图片标号', min_value=1, max_value=len(img_names), step=1))
                img_names = os.listdir(test_path)
                # 按照数字顺序排序
                img_names.sort(key=lambda x: int(x.split('.')[0]))
                # img_names = img
            except:
                pass
#                 st.error('公开链接无法获取本地文件，请切换为服务器运行')

    if file_bytes is not None:
        image = np.array(bytearray(file_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, channels='BGR', width=300)
        if st.button('预测'):
            st.spinner('预测中...')
            pred = predict(image)
            st.write('预测结果：', cla_dict[pred.item()])
            st.balloons()
    elif env == '本地运行':
        try:
            st.write(os.path.join(test_path, img_names[num - 1]))
            image = read_img(os.path.join(test_path, img_names[num - 1]))
            st.image(image, channels='BGR', width=300)
            if st.button('预测'):
                st.spinner('预测中...')
                pred = predict(image)
                st.write('预测结果：', cla_dict[pred.item()])
                st.balloons()
        except:
            st.error('公开链接无法获取本地文件，请切换为服务器运行')
    # st.balloons(size=30)

if __name__ == '__main__':
    main()

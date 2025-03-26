import pandas as pd
import streamlit as st

from src.dataset_generation import generation_dataset_1, generation_dataset_2
from src.dataset_review import data_review

st.set_page_config(page_title="LLM生成与审核训练数据", page_icon="∰", layout="wide")


sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''
st.markdown(sysmenu, unsafe_allow_html=True)

st.header("LLM生成与审核训练数据")
prompt_input = st.text_area('要生成训练数据的prompt', height=200)

if st.button('生成'):
    if prompt_input is not None or len(prompt_input.strip()) == 0:
        generation_dataset_1(prompt_input, 'data/generation_dataset.xlsx')
        # generation_dataset_2(prompt_input, 'data/generation_dataset.xlsx', 'data/dataset.xlsx')
        st.write(pd.read_excel('data/dataset.xlsx'))

prompt_review_input = st.text_area('对LLM生成的训练数据进行审核prompt', height=200)
if st.button('审核'):
    if prompt_review_input is not None or len(prompt_input.strip()) == 0:
        data_review(prompt_review_input, 'data/dataset.xlsx', 'data/result_r1.xlsx')
        st.write(pd.read_excel('data/result_r1.xlsx'))






import gradio as gr
from fastai.vision.all import *

path = Path('model.pkl')
learn_inf = load_learner(path)

def img(input_img):
    input_img = PILImage.create(input_img)
    rc, _, _ = learn_inf.predict(input_img)
    return rc

app = gr.Interface(fn=img, inputs=gr.Image(type='filepath'), outputs='text')
app.launch()

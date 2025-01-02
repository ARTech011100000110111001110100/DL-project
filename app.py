import gradio as gr
from fastai.vision.all import *
import webbrowser as wb

path = Path('model.pkl')
learn_inf = load_learner(path)
url = 'https://www.youtube.com/watch?v=_HCqOqGwDfE'

def img(input_img):
    input_img = PILImage.create(input_img)
    rc, _, _ = learn_inf.predict(input_img)
    if rc == 'black':
        wb.open(url)
    return rc

app = gr.Interface(fn=img, inputs=gr.Image(type='filepath'), outputs='text')
app.launch()

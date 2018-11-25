# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import tkinter
from model import lenet
from ui import mnistCanvas
from inference import inference
from PIL import Image, ImageDraw
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def main():
    root = tkinter.Tk()
    root.geometry('300x400')
    frame = tkinter.Frame(root, width=256, height=256)
    frame.pack_propagate(0)
    frame.pack(side='top')
    canvas1 = mnistCanvas(frame)
    infer = inference()

    # mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    # im = mnist.train.images[1]
    # im = im.reshape(-1, 28)
    # plt.imshow(im)
    # plt.show()


    def test_click():
        img = canvas1.image1
        result = infer.predict(img)
        result = int(result)
        canvas1.canvas.delete('all')
        canvas1.image1 = Image.new('RGB', (256, 256), 'black')
        canvas1.draw = ImageDraw.Draw(canvas1.image1)
        label2['text'] = str(result)
    
    botton_Inference = tkinter.Button(root, text='Inference', width=7, height=1, command=test_click)
    botton_Inference.pack()
    label1 = tkinter.Label(root, justify='center', text='Inference result is')
    label1.pack()
    label2 = tkinter.Label(root, justify='center')
    label2['font'] = ('Arial, 38')
    label2.pack()
    root.mainloop()
    

if __name__ == '__main__':
    main()
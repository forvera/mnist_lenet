# -*- coding:utf-8 -*-
import tkinter
from PIL import Image, ImageDraw

class mnistCanvas:
    def __init__(self, root):
        self.root = root
        self.canvas = tkinter.Canvas(root, width=256, height=256, bg='black')
        self.canvas.pack()
        self.image1 = Image.new('RGB', (256, 256), 'black')
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>', self.Draw)

    def Draw(self, event):
        self.canvas.create_oval(event.x, event.y, event.x, event.y, outline="white", width=20)
        self.draw.ellipse((event.x-10, event.y-10, event.x+10, event.y+10), fill=(255, 255, 255))
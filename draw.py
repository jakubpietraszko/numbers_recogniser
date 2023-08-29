import tkinter as tk
from tkinter import ttk
import PIL
from PIL import Image, ImageDraw
import torch
from torch import nn
from torchvision import transforms
from model import Model


class Draw():
    def __init__(self) -> None:
        self.root: tk.Tk = tk.Tk()
        self.root.title('Test Program')
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_frame: ttk.Frame = ttk.Frame(self.root,
                                               padding='3 3 3 3')
        self.main_frame.grid()
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.drawing: tk.Canvas = tk.Canvas(self.main_frame,
                                            bg='black',
                                            width=400,
                                            height=400,
                                            borderwidth=2,
                                            relief='raised')
        self.drawing.grid(column=0,
                          row=0,
                          columnspan=3,
                          rowspan=3)
        self.drawing.grid_rowconfigure(0, weight=1)
        self.drawing.grid_columnconfigure(0, weight=1)
        self.drawing.bind('<B1-Motion>',
                          self.draw)
        self.drawing.bind('<B3-Motion>',
                          self.erase)
        self.drawing.bind('<ButtonRelease-1>',
                          self.reset)
        self.drawing.bind('<ButtonRelease-3>',
                          self.reset)

        self.IMAGE: PIL.Image = PIL.Image.new('L', (400, 400), 'black')
        self.DRAW: ImageDraw = ImageDraw.Draw(self.IMAGE)

        self.last_x: int = None
        self.last_y: int = None

        self.button: ttk.Button = ttk.Button(self.main_frame,
                                             text='check',
                                             command=self.check)
        self.button.grid(column=1,
                         row=3)
        self.button.grid_rowconfigure(0, weight=1)

        self.button.grid_columnconfigure(0, weight=1)

        self.number_returned: tk.StringVar = tk.StringVar()

        self.text: ttk.Label = ttk.Label(self.main_frame,
                                         text='returned number',
                                         borderwidth=2,
                                         relief='groove')
        self.text.grid(column=3,
                       row=0,
                       rowspan=2)

        self.answear: ttk.Label = ttk.Label(self.main_frame,
                                            textvariable=self.number_returned,
                                            borderwidth=2,
                                            relief='groove',
                                            width=1)
        self.answear.grid(column=3,
                          row=2)

        for child in self.main_frame.winfo_children():
            child.grid_configure(padx=5,
                                 pady=5)

        self.root.mainloop()

    def check(self, file_name: str = 'image') -> None:
        self.save(self.drawing, 'image')

        model: nn.Module = torch.load(f='models/model')

        img: Image = Image.open('image_compressed.png')

        convert: transforms.ToTensor = transforms.ToTensor()
        image: torch.Tensor = convert(img)
        image: torch.Tensor = image.unsqueeze(dim=0)

        model: nn.Module = Model(input_shape=1,
                                 hidden_units=10,
                                 output_shape=10)
        model.load_state_dict(torch.load(f='models/model'))

        num_to_return: int = model(image).softmax(dim=1).argmax(dim=1).item()

        self.number_returned.set(num_to_return)

    def draw(self, event: tk.Event) -> None:
        x: int = event.x
        y: int = event.y

        if self.last_x and self.last_y:
            self.drawing.create_line((self.last_x, self.last_y, x, y),
                                     width=10,
                                     fill='white')
            self.DRAW.line((self.last_x, self.last_y, x, y),
                           fill='white',
                           width=10)

        self.last_x, self.last_y = x, y

    def erase(self, event: tk.Event) -> None:
        x: int = event.x
        y: int = event.y

        if self.last_x and self.last_y:
            self.drawing.create_line((self.last_x, self.last_y, x, y),
                                     width=50,
                                     fill='black')
            self.DRAW.line((self.last_x, self.last_y, x, y),
                           fill='black',
                           width=50)

        self.last_x, self.last_y = x, y

    def reset(self, event: tk.Event) -> None:
        self.last_x, self.last_y = None, None

    def save(self, canvas: tk.Canvas, filename: str = 'image') -> None:
        self.IMAGE.save(filename + '.png')

        img: Image = Image.open(filename + '.png')

        compressed_img: Image = img.resize((28, 28))
        compressed_img.save(filename + '_compressed.png')

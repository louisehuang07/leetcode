import flet as ft
import cv2
from ImageProcessor import *

def main(page: ft.Page):
    app = ImageProcessorApp(page)
    app.build_ui()


ft.app(target=main)

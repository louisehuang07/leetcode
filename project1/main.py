import flet as ft
import cv2
import numpy as np 
import base64
import time  # Import time module to measure execution time
from utils import *



def to_base64(image):
    base64_image = cv2.imencode('.png', image)[1]
    base64_image = base64.b64encode(base64_image).decode('utf-8') 
    return base64_image

def main(page):
    # Create a blank image for the initial display,
    # image element does not support None for src_base64
    init_image = np.zeros((480, 640, 3), dtype=np.uint8) + 128
    init_base64_image = to_base64(init_image)

    # image_src = ft.Image(src_base64=init_base64_image, width=640, height=480)
    image_src = ft.Image(
        src_base64=init_base64_image, 
        fit=ft.ImageFit.FILL    # https://flet.dev/docs/controls/image/
    )
    # image_src = ft.Image(src_base64=init_base64_image, width=640, height=480)

    # image_row = ft.Row([image_src, image_src])

    image = None
    # image= None 

    original_image = None  # To store the original image for reset
    
    scale_factor = 1.0  # original scale = 1.0
    scale_factor_manual = 1.0


    time_records = []  # List to store time records
    
    appbar_text_ref = ft.Ref[ft.Text]()
    
    def info(e):
        print(f"{e.control.content.value}.on_click")
        page.open(ft.SnackBar(content=ft.Text(f"{e.control.content.value} was clicked!")))
        appbar_text_ref.current.value = e.control.content.value
        page.update()


    def save_image(e):
        print(f"{e.control.content.value}.on_click")
        page.open(ft.SnackBar(content=ft.Text(f"{e.control.content.value} was clicked!")))
        appbar_text_ref.current.value = e.control.content.value
        page.update()


    def close_gui(e):
        print(f"{e.control.content.value}.on_click")
        page.open(ft.SnackBar(content=ft.Text(f"{e.control.content.value} was clicked!")))
        appbar_text_ref.current.value = e.control.content.value
        page.update()


    def handle_submenu_open(e):
        print(f"{e.control.content.value}.on_open")


    def handle_submenu_close(e):
        print(f"{e.control.content.value}.on_close")


    def handle_submenu_hover(e):
        print(f"{e.control.content.value}.on_hover")


    def update_scaled_image():
        nonlocal original_image, scale_factor
        if original_image is None:
            return
        # get new size
        new_width = int(original_image.shape[1] * scale_factor)
        new_height = int(original_image.shape[0] * scale_factor)
        start_time = time.time()
        scaled_image = cv2.resize(original_image, (new_width, new_height))
        image_src.src_base64 = to_base64(scaled_image)
        image_src.update()
        end_time = time.time()
        add_time_record("Update Scale", end_time - start_time)


    def update_scaled_image_manually():
        nonlocal original_image, scale_factor_manual
        if original_image is None:
            return
        # get new size
        new_width = int(original_image.shape[1] * scale_factor_manual)
        new_height = int(original_image.shape[0] * scale_factor_manual)

        # TODO replace with manually resize
        # scaled_image = cv2.resize(original_image, (new_width, new_height))
        start_time = time.time()
        scaled_image = resize(original_image, new_height, new_width)
        image_src.src_base64 = to_base64(scaled_image)
        image_src.update()
        end_time = time.time()
        add_time_record("Update Scale Manually", end_time - start_time)
    

    def scale_up(e):
        nonlocal scale_factor
        scale_factor += 0.1
        update_scaled_image()

    def scale_down(e):
        nonlocal scale_factor
        if scale_factor > 0.1:  # 确保比例不会小于 0.1
            scale_factor -= 0.1
        update_scaled_image()

    
    


    def rotate_image(e):
        nonlocal image
        if image is None:
            return
        start_time = time.time()
        # Rotate the image by 90 degrees clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_src.src_base64 = to_base64(image)
        image_src.update()
        end_time = time.time()
        add_time_record("Rotate 90°", end_time - start_time)
    

    def rotate_image_manual(e):
        nonlocal image
        if image is None:
            return
        start_time = time.time()
        
        # Rotate the image by 90 degrees clockwise
        # image= cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image= rotate(image)
        # image= rotate_1(image)

        image_src.src_base64 = to_base64(image)
        image_src.update()
        end_time = time.time()
        add_time_record("Rotate 90° Manually", end_time - start_time)


    def invert_colors(e):
        nonlocal image
        if image is None:
            return
        start_time = time.time()
        # Invert the colors of the image (negative effect)
        image = cv2.bitwise_not(image)  # Update image with inverted colors
        image_src.src_base64 = to_base64(image)
        image_src.update()
        end_time = time.time()
        add_time_record("Invert Colors", end_time - start_time)


    def invert_colors_manual(e):
        nonlocal image
        if image is None:
            return
        start_time = time.time()
        # Invert the colors of the image (negative effect)
        # image= cv2.bitwise_not(image)  # Update image with inverted colors
        image= bitwise_not(image)
        
        image_src.src_base64 = to_base64(image)
        image_src.update()
        end_time = time.time()
        add_time_record("Invert Colors Manually", end_time - start_time)


    def grayscale_image(e):
        nonlocal image
        if image is None:
            return
        start_time = time.time()
        # Convert the image to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert the grayscale image back to BGR (3 channels) for consistency
        grayscale_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        image = grayscale_bgr  # Update image with grayscale
        image_src.src_base64 = to_base64(image)
        image_src.update()
        end_time = time.time()
        add_time_record("Grayscale", end_time - start_time)


    def grayscale_image_manual(e):
        nonlocal image
        if image is None:
            return
        start_time = time.time()
        # # Convert the image to grayscale
        # grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # # Convert the grayscale image back to BGR (3 channels) for consistency
        # grayscale_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        grayscale_bgr = grayscale(image)
        
        image= grayscale_bgr  # Update image with grayscale
        image_src.src_base64 = to_base64(image)
        image_src.update()
        end_time = time.time()
        add_time_record("Grayscale Manually", end_time - start_time)


    def scale_up_manual(e):
        nonlocal scale_factor_manual
        scale_factor_manual += 0.1
        update_scaled_image_manually()


    def scale_down_manual(e):
        nonlocal scale_factor_manual
        if scale_factor_manual > 0.1:  
            scale_factor_manual -= 0.1
        update_scaled_image_manually()


    def reset_to_original(e):
        nonlocal image, image, original_image
        if original_image is None:
            return
        # start_time = time.time()
        # Restore the original image size
        image = original_image
        # image= original_image

        image_src.src_base64 = to_base64(image)
        image_src.update()
        # image_src.src_base64 = to_base64(image)
        # image_src.update()

        # end_time = time.time()
        # add_time_record("Reset to Original Size", end_time - start_time)

    def on_file_selected(e):
        nonlocal image, image, original_image
        file_path = e.files[0].path
        # print("file selected :", file_path)
        original_image = cv2.imread(file_path)
        image = original_image.copy()  # Store a copy of the original image
        image= original_image.copy()

        base64_image = to_base64(image)

        # Display loaded image in both image_src and image_src
        image_src.src_base64 = base64_image
        image_src.update()


    def add_time_record(operation_name, time_taken):
        """Function to add the operation time to the record list."""
        if len(time_records) >= 10:
            time_records.pop(0)  # Remove the oldest record if there are already 10
        time_records.append(f"{operation_name}: {time_taken:.4f} seconds")
        time_text = "\n".join(time_records)
        time_display.value = f"Last 10 operation times:\n{time_text}"
        time_display.update()


    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)

    def on_click(e):
        file_picker.pick_files(allow_multiple=False, 
                               file_type=ft.FilePickerFileType.IMAGE)
    time_display = ft.Text(
        value="Last 10 operation times:\n", 
        size=16, 
        style=ft.TextStyle(
            font_family="Consolas",  
            # font_weight=ft.FontWeight.BOLD 
            )
        )
    # Images and time display layout
    image_row = ft.Column(
        [
            ft.Row(
                [image_src],
                alignment=ft.MainAxisAlignment.CENTER,
                ),  # Display images side by side
            ft.Row(
                [time_display],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,  # Vertically center the images
    )
    # image_row = ft.Column(
    #     [
    #         image_src,  # Image at the top
    #         time_display,  # Text at the bottom
    #     ],
    #     alignment=ft.MainAxisAlignment.CENTER,  # Vertically center the column's content
    #     # horizontal_alignment=ft.VerticalAlignment.CENTER  # Align items vertically in the center
    # )

    # # layout
    # main_layout = ft.Row(
    #     [button_column, image_row],
    #     alignment=ft.MainAxisAlignment.START,
    # )
    # page.window_maximized = True  # Set the window to open in full screen
    
    page.window_width = 1200  # 添加额外的空间（如边框或控件）
    page.window_height = 1200
    
    # Your UI 
    page.appbar=ft.AppBar(
        leading=ft.Icon(ft.icons.PALETTE),
        leading_width=40,
        title=ft.Text("Image Viewer for Machine Vision."),
        center_title=False,
        bgcolor=ft.colors.SURFACE_VARIANT,
        actions=[
            ft.IconButton(ft.icons.WB_SUNNY_OUTLINED, on_click=on_file_selected),
            ft.IconButton(ft.icons.PENDING_ACTIONS),
            ft.PopupMenuButton(
                items=[
                    ft.PopupMenuItem(
                        text="Item", 
                        on_click=info
                        ),
                ]
            ),
        ]
    )

    menubar = ft.MenuBar(
        expand=True,
        style=ft.MenuStyle(
            alignment=ft.alignment.top_left,
            bgcolor=ft.colors.GREEN_200,
            mouse_cursor={
                ft.ControlState.HOVERED: ft.MouseCursor.WAIT,
                ft.ControlState.DEFAULT: ft.MouseCursor.ZOOM_OUT,
            },
        ),
        controls=[
            ft.SubmenuButton(   # open files
                content=ft.Text("File"),
                on_open=handle_submenu_open,
                on_close=handle_submenu_close,
                on_hover=handle_submenu_hover,
                controls=[
                    ft.MenuItemButton(
                        content=ft.Text("Open Image"),
                        leading=ft.Icon(ft.icons.FILE_OPEN_ROUNDED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        on_click=on_click,
                    ),
                    # ft.MenuItemButton(
                    #     content=ft.Text("About"),
                    #     leading=ft.Icon(ft.icons.INFO),
                    #     style=ft.ButtonStyle(
                    #         bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                    #     ),
                    #     on_click=info,
                    # ),
                    ft.MenuItemButton(
                        content=ft.Text("Save"),
                        leading=ft.Icon(ft.icons.SAVE),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        on_click=save_image,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Quit"),
                        leading=ft.Icon(ft.icons.CLOSE),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        on_click=close_gui,
                    ),
                ],
            ),
            ft.SubmenuButton(   # process images(project 1)
                content=ft.Text("Opencv"),
                on_open=handle_submenu_open,
                on_close=handle_submenu_close,
                on_hover=handle_submenu_hover,
                controls=[
                    ft.MenuItemButton(
                        content=ft.Text("Rotate 90°"),
                        leading=ft.Icon(ft.icons.CROP_ROTATE_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=rotate_image,
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Zoom"),
                        leading=ft.Icon(ft.icons.ZOOM_OUT_MAP),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("+10%"),
                                leading=ft.Icon(ft.icons.ZOOM_OUT_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=scale_up,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("-10%"),
                                leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=scale_down,
                            ),
                        ],
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Invert"),
                        leading=ft.Icon(ft.icons.INVERT_COLORS_SHARP),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=invert_colors,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Grayscale"),
                        leading=ft.Icon(ft.icons.COLOR_LENS_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=grayscale_image,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        leading=ft.Icon(ft.icons.RESET_TV_ROUNDED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=reset_to_original,
                    )
                ],
            ),
            ft.SubmenuButton(   # reinplement to process images(project 1)
                content=ft.Text("Developing"),
                on_open=handle_submenu_open,
                on_close=handle_submenu_close,
                on_hover=handle_submenu_hover,
                controls=[
                    ft.MenuItemButton(
                        content=ft.Text("Rotate 90°"),
                        leading=ft.Icon(ft.icons.CROP_ROTATE_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=rotate_image_manual,
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Zoom"),
                        leading=ft.Icon(ft.icons.ZOOM_OUT_MAP),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("+10%"),
                                leading=ft.Icon(ft.icons.ZOOM_OUT_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=scale_up_manual,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("-10%"),
                                leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=scale_down_manual,
                            ),
                        ],
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Invert"),
                        leading=ft.Icon(ft.icons.INVERT_COLORS_SHARP),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=invert_colors_manual,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Grayscale"),
                        leading=ft.Icon(ft.icons.COLOR_LENS_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=grayscale_image_manual,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        leading=ft.Icon(ft.icons.RESET_TV_ROUNDED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=reset_to_original,
                    )
                ],
            )
        ],
    )
    page.scroll = "adaptive"
    page.add(ft.Row([menubar]))


    # page.add(main_layout)
    page.add(image_row)

ft.app(target=main)

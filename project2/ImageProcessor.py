import math
import flet as ft
import cv2
import numpy as np
import base64
import time  # Import time module to measure execution time
from image_operator import *
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu


class ImageProcessorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.init_image = np.zeros((480, 640, 3), dtype=np.uint8) + 128
        self.init_base64_image = self.to_base64(self.init_image)

        self.image_src = ft.Image(
            src_base64=self.init_base64_image,
            fit=ft.ImageFit.FILL
        )

        self.image = None
        self.original_image = None
        self.scale_factor = 1.0 # for zoom in and zoom out
        self.shear_angle = 10 # for bilinear shear
        self.time_records = []

        self.time_display = ft.Text(
            value="Last 1 operation times:\n",
            size=16,
            style=ft.TextStyle(font_family="Consolas"),
        )

        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.page.overlay.append(self.file_picker)
    
    def build_ui(self):
        appbar=ft.AppBar(
        leading=ft.Icon(ft.icons.PALETTE),
        leading_width=40,
        title=ft.Text(value="Image Viewer for Machine Vision.",
                      style=ft.TextStyle(font_family="Consolas")),
        
        center_title=False,
        bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100},
        actions=[
            ft.IconButton(ft.icons.PERM_MEDIA, on_click=lambda e: self.file_picker.pick_files(
                                allow_multiple=False,
                                file_type=ft.FilePickerFileType.IMAGE,  # icons.FILE_OPEN_ROUNDED
                            )),
            ft.IconButton(ft.icons.CLOSE,on_click=lambda e: self.page.window_close()),
        ]
    )
        # Menu bar
        menubar = ft.MenuBar(
            expand=True,
            controls=[
                ft.SubmenuButton(
                    content=ft.Text("Edit"),leading=ft.Icon(ft.icons.EDIT),
                    controls=[
                        ft.MenuItemButton(
                        content=ft.Text("Rotate 90°"),
                        leading=ft.Icon(ft.icons.CROP_ROTATE_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.rotate_image,
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Zoom"),
                        leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("+10%"),
                                leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.scale_up,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("-10%"),
                                leading=ft.Icon(ft.icons.ZOOM_OUT_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.scale_down,
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
                        on_click=self.invert_colors,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Grayscale"),
                        leading=ft.Icon(ft.icons.COLOR_LENS_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.grayscale_image,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        # leading=ft.Icon(ft.icons.RESET_TV_ROUNDED),
                        leading=ft.Icon(ft.icons.EDIT_OFF),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.reset_to_original,
                    )
                    ],
                ),
                ft.SubmenuButton(
                    content=ft.Text("Edit"),
                    leading=ft.Icon(ft.icons.NUMBERS),
                    controls=[
                        ft.MenuItemButton(
                        content=ft.Text("Rotate 90°"),
                        leading=ft.Icon(ft.icons.CROP_ROTATE_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.rotate_image_,
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
                                leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.scale_up_,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("-10%"),
                                leading=ft.Icon(ft.icons.ZOOM_OUT_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.scale_down_,
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
                        on_click=self.invert_colors_,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Grayscale"),
                        leading=ft.Icon(ft.icons.COLOR_LENS_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.grayscale_image_,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        leading=ft.Icon(ft.icons.FILTER_ALT_OFF),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.reset_to_original,
                    )
                    ],
                ),
                # Filter 
                ft.SubmenuButton(
                    content=ft.Text("Filter"),
                    leading=ft.Icon(ft.icons.FILTER_ALT),
                    controls=[
                    ft.SubmenuButton(   
                        content=ft.Text("Bilinear Interpolation"),
                        leading=ft.Icon(ft.icons.DATA_USAGE_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("Zoom In"),
                                leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.scale_up,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Shear Up"),
                                leading=ft.Icon(ft.icons.ARROW_UPWARD),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.shear_1_up,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Shear Down"),
                                leading=ft.Icon(ft.icons.ARROW_DOWNWARD),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.shear_1_down,
                            ),
                            # ft.MenuItemButton(
                            #     content=ft.Text("Shear_2_UP"),
                            #     leading=ft.Icon(ft.icons.CONTENT_CUT),
                            #     close_on_click=False,
                            #     style=ft.ButtonStyle(
                            #         bgcolor={
                            #             ft.ControlState.HOVERED: ft.colors.PURPLE_200
                            #         }
                            #     ),
                            #     on_click=self.shear_2_up,
                            # ),
                            # ft.MenuItemButton(
                            #     content=ft.Text("Shear_2_DOWN"),
                            #     leading=ft.Icon(ft.icons.CONTENT_CUT),
                            #     close_on_click=False,
                            #     style=ft.ButtonStyle(
                            #         bgcolor={
                            #             ft.ControlState.HOVERED: ft.colors.PURPLE_200
                            #         }
                            #     ),
                            #     on_click=self.shear_2_down,
                            # ),
                            ft.MenuItemButton(
                                content=ft.Text("Perspective"),
                                leading=ft.Icon(ft.icons.GRID_ON),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.bilinear_perspective,
                            ),
                        ],
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Histogram"),
                        leading=ft.Icon(ft.icons.BAR_CHART),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("Histogram Plot"),
                                leading=ft.Icon(ft.icons.BAR_CHART),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.histogram_plot,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Histogram Equalization"),
                                leading=ft.Icon(ft.icons.CONTRAST_OUTLINED),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.histogram_equalization,
                            ),
                        ],
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Add Noise"),
                        leading=ft.Icon(ft.icons.NOW_WALLPAPER),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("Gaussian"),
                                leading=ft.Icon(ft.icons.GAMEPAD),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.add_gaussian,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Salt Pepper"),
                                leading=ft.Icon(ft.icons.VOICEMAIL),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.add_salt_pepper,
                            ),
                        ],
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Median Filter"),
                        leading=ft.Icon(ft.icons.FILTER_B_AND_W),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.median,
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Sharpen"),
                        leading=ft.Icon(ft.icons.BRIGHTNESS_3_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("Robert Opt"),
                                leading=ft.Icon(ft.icons.FILTER_1),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.sharpen_robert,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Sobel Opt"),
                                leading=ft.Icon(ft.icons.FILTER_2),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.sharpen_sobel,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Laplace Opt"),
                                leading=ft.Icon(ft.icons.FILTER_3),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.sharpen_laplace,
                            ),
                        ],
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Frequency Domain"),
                        leading=ft.Icon(ft.icons.WINDOW),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            
                            ft.MenuItemButton(
                                content=ft.Text("Ideal High-pass Filter"),
                                leading=ft.Icon(ft.icons.TRENDING_UP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.frequency_high_pass_ideal,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Butterworth High-pass Filter"),
                                leading=ft.Icon(ft.icons.TRENDING_UP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.frequency_high_pass_butterworth,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Gaussian High-pass Filter"),
                                leading=ft.Icon(ft.icons.TRENDING_UP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.frequency_high_pass_gaussian,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Ideal Low-pass Filter"),
                                leading=ft.Icon(ft.icons.TRENDING_DOWN),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.frequency_low_pass_ideal,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Butterworth Low-pass Filter"),
                                leading=ft.Icon(ft.icons.TRENDING_DOWN),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.frequency_low_pass_butterworth,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Gaussian Low-pass Filter"),
                                leading=ft.Icon(ft.icons.TRENDING_DOWN),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.frequency_low_pass_gaussian,
                            ),
                        ],
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        leading=ft.Icon(ft.icons.FILTER_ALT_OFF),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.reset_to_original,
                    )
                    ],
                ),
                # Filter 
                ft.SubmenuButton(
                    content=ft.Text("Filter"),
                    leading=ft.Icon(ft.icons.NUMBERS),
                    controls=[
                    ft.SubmenuButton(   
                        content=ft.Text("Bilinear Interpolation"),
                        leading=ft.Icon(ft.icons.DATA_USAGE_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("Zoom In"),
                                leading=ft.Icon(ft.icons.ZOOM_IN_MAP),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.scale_up_,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Shear Up"),
                                leading=ft.Icon(ft.icons.ARROW_UPWARD),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.shear_up_m,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Shear Down"),
                                leading=ft.Icon(ft.icons.ARROW_DOWNWARD),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.shear_down_m,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Perspective"),
                                leading=ft.Icon(ft.icons.GRID_ON),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.bilinear_perspective_m,
                            ),
                        ],
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Histogram"),
                        leading=ft.Icon(ft.icons.BAR_CHART),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.histogram_m,
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Median Filter"),
                        leading=ft.Icon(ft.icons.FILTER_B_AND_W),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.median_m,
                    ),
                    ft.SubmenuButton(
                        content=ft.Text("Sharpen"),
                        leading=ft.Icon(ft.icons.BRIGHTNESS_3_OUTLINED),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        controls=[
                            ft.MenuItemButton(
                                content=ft.Text("Robert Opt"),
                                leading=ft.Icon(ft.icons.FILTER_1),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.sharpen_robert_m,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Sobel Opt"),
                                leading=ft.Icon(ft.icons.FILTER_2),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.sharpen_sobel_m,
                            ),
                            ft.MenuItemButton(
                                content=ft.Text("Laplace Opt"),
                                leading=ft.Icon(ft.icons.FILTER_3),
                                close_on_click=False,
                                style=ft.ButtonStyle(
                                    bgcolor={
                                        ft.ControlState.HOVERED: ft.colors.PURPLE_200
                                    }
                                ),
                                on_click=self.sharpen_laplace_m,
                            ),
                        ],
                    ),
                    # ft.SubmenuButton(
                    #     content=ft.Text("Frequency Domain"),
                    #     leading=ft.Icon(ft.icons.WAVES),
                    #     style=ft.ButtonStyle(
                    #         bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                    #     ),
                    #     controls=[
                    #         ft.MenuItemButton(
                    #             content=ft.Text("High pass"),
                    #             leading=ft.Icon(ft.icons.TRENDING_UP),
                    #             close_on_click=False,
                    #             style=ft.ButtonStyle(
                    #                 bgcolor={
                    #                     ft.ControlState.HOVERED: ft.colors.PURPLE_200
                    #                 }
                    #             ),
                    #             on_click=self.frequency_high_pass_m,
                    #         ),
                    #         ft.MenuItemButton(
                    #             content=ft.Text("Low pass"),
                    #             leading=ft.Icon(ft.icons.TRENDING_DOWN),
                    #             close_on_click=False,
                    #             style=ft.ButtonStyle(
                    #                 bgcolor={
                    #                     ft.ControlState.HOVERED: ft.colors.PURPLE_200
                    #                 }
                    #             ),
                    #             on_click=self.frequency_low_pass_m,
                    #         ),
                    #     ],
                    # ),
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        leading=ft.Icon(ft.icons.FILTER_ALT_OFF),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.reset_to_original,
                    )
                    ],
                ),
                # Segment
                ft.SubmenuButton(
                    content=ft.Text("Segment"),
                    leading=ft.Icon(ft.icons.NUMBERS),
                    controls=[
                    # dual-threshold
                    ft.MenuItemButton(
                        content=ft.Text("Dual Threshold Segmentation"),
                        leading=ft.Icon(ft.icons.DATA_THRESHOLDING),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.dual_threshold_segmentation,
                    ),
                    # region-growing
                    ft.MenuItemButton(
                        content=ft.Text("Region Growing Segmentation"),
                        leading=ft.Icon(ft.icons.KEYBOARD_DOUBLE_ARROW_UP),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.region_growing,
                    ),
                    # otsu 
                    ft.MenuItemButton(
                        content=ft.Text("otsu Segmentation"),
                        leading=ft.Icon(ft.icons.FORMAT_QUOTE),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.otsu_segmentation,
                    ),
                    # watershed
                    ft.MenuItemButton(
                        content=ft.Text("Watershed Segmentation"),
                        leading=ft.Icon(ft.icons.WATER),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.watershed_segmentation,
                    ),
                    
                    ft.MenuItemButton(
                        content=ft.Text("Reset"),
                        leading=ft.Icon(ft.icons.FILTER_ALT_OFF),
                        style=ft.ButtonStyle(
                            bgcolor={ft.ControlState.HOVERED: ft.colors.GREEN_100}
                        ),
                        close_on_click=False,
                        on_click=self.reset_to_original,
                    )
                    ],
                ),
            ],
        )

        # Layout
        image_row = ft.Column(
            [
                ft.Row([self.image_src], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([self.time_display], alignment=ft.MainAxisAlignment.CENTER),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )

        self.page.add(appbar)
        self.page.add(ft.Row([menubar]))
        self.page.add(image_row)
    
    # 1. Dual Threshold
    def dual_threshold_segmentation(self,e):
        check_if_image_exist(self.image)
        start_time = time.time()
        low_thresh = 190
        high_thresh = 255
        image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # segmented = np.zeros_like(image, dtype=np.uint8)
        # segmented[(image >= low_thresh) & (image <= high_thresh)] = 255

        # # make sure output image is 0 or 255
        # segmented = (segmented > 0).astype(np.uint8) * 255  

        # # broadcast
        # result = cv2.bitwise_and(image, segmented)  
        
        _, thresholded_img = cv2.threshold(image, low_thresh, high_thresh, cv2.THRESH_BINARY)
 
        # images_combined = np.hstack((image, thresholded_img))
        result = cv2.bitwise_and(image, thresholded_img)  

        # return nose_mask
        self.image_src.src_base64 = self.to_base64(result)
        self.image_src.update()
        self.add_time_record("Dual Threshold Segmentation", time.time() - start_time)

    # 2. Region_Growing (blimp = 748, 642)
    def region_growing(self,e):
        # image, seed_point, threshold=15
        check_if_image_exist(self.image)
        start_time = time.time()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # size of the grayscale
        h, w = gray.shape
        threshold = 85

        # output image
        outimg  = np.zeros_like(gray, dtype=np.uint8)

        # seed point, starting point of growth
        seed = (h//2, w//2)   # half of the height and width

        # Gray value of the seed point
        seed_value = gray[seed]
        seed_points  = [seed]

        # processed mark
        processed = np.zeros_like(gray, dtype=np.bool_) 


        while seed_points :
            # current seed point
            x, y = seed_points .pop(0)

            processed[x, y] = True
            outimg[x, y] = 255  


            for xn, yn in get8n(x, y, gray.shape):
                if not processed[xn, yn]:  # if the neighbors have not been processed
                    # if this is similar
                    if abs(int(gray[xn, yn]) - int(seed_value)) <= threshold:
                        seed_points.append((xn, yn))  # add to seed list
                    processed[xn, yn] = True  # mark as processed
        
        # make sure output image is 0 or 255
        outimg = (outimg > 0).astype(np.uint8) * 255  

        # broadcast
        result = cv2.bitwise_and(gray, outimg)  

        # outimg = outimg * gray
        # return mask
        self.image_src.src_base64 = self.to_base64(result)
        self.image_src.update()
        self.add_time_record("Region Growing Segmentation", time.time() - start_time)

    # 3. Otsu 
    def otsu_segmentation(self,e):
        check_if_image_exist(self.image)
        start_time = time.time()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = threshold_otsu(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    

        # make sure output image is 0 or 255
        binary = (binary > 0).astype(np.uint8) * 255  

        # broadcast
        result = cv2.bitwise_and(gray, binary)  

        self.image_src.src_base64 = self.to_base64(result)
        self.image_src.update()
        self.add_time_record("OTSU Segmentation", time.time() - start_time)

    # 4. Watershed
    def watershed_segmentation(self,e):
        check_if_image_exist(self.image)
        start_time = time.time()
        image = self.image
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        equalized = cv2.equalizeHist(blurred)
        
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 7, 2)
        
        kernel = np.ones((2, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.001 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        
        markers = markers + 1
        
        markers[unknown == 255] = 0
        markers = cv2.watershed(self.image, markers)
        image[markers == -1] = [255, 0, 0]

        # make sure output image is 0 or 255
        binary = (markers > 0).astype(np.uint8) * 255  

        # broadcast
        result = cv2.bitwise_and(gray, binary)  

        # return segmentation
        self.image_src.src_base64 = self.to_base64(result)
        self.image_src.update()
        self.add_time_record("Watershed Segmentation", time.time() - start_time)

    
    def to_base64(self, image):
        base64_image = cv2.imencode('.png', image)[1]
        return base64.b64encode(base64_image).decode('utf-8')

    def add_time_record(self, operation_name, time_taken):
        if len(self.time_records) >= 1:
            self.time_records.pop(0)
        self.time_records.append(f"{operation_name}: {time_taken:.4f} seconds")
        time_text = "\n".join(self.time_records)
        self.time_display.value = f"Last 1 operation times:\n{time_text}"
        self.time_display.update()


    def on_file_selected(self, e):
        file_path = e.files[0].path
        self.original_image = cv2.imread(file_path)
        self.image = self.original_image.copy()
        base64_image = self.to_base64(self.image)
        self.image_src.src_base64 = base64_image
        self.image_src.update()
    
    def update_scaled_image(self):
        check_if_image_exist(self.original_image)
        new_width = int(self.original_image.shape[1] * self.scale_factor)
        new_height = int(self.original_image.shape[0] * self.scale_factor)
        start_time = time.time()

        self.image = cv2.resize(self.original_image, (new_width, new_height))
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Update Scale", time.time() - start_time)

    def scale_up(self, e):
        self.scale_factor += 0.1
        self.update_scaled_image()

    def scale_down(self, e):
        if self.scale_factor > 0.1:
            self.scale_factor -= 0.1
        self.update_scaled_image()

    def rotate_image(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Rotate 90°", time.time() - start_time)

    def invert_colors(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        self.image = cv2.bitwise_not(self.image)
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Invert Colors", time.time() - start_time)

    def grayscale_image(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grayscale_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        # print('image', self.image.shape)    # h,w,3
        # print('grayscale', grayscale.shape) # h,w
        # print('grayscale_bgr', grayscale_bgr.shape) # h,w,3

        self.image = grayscale_bgr
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Grayscale", time.time() - start_time)

    def reset_to_original(self, e):
        check_if_image_exist(self.original_image)
        self.image = self.original_image
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()

    # Manually marked as '_'
    def update_scaled_image_(self):
        check_if_image_exist(self.original_image)
        
        start_time = time.time()

        self.image = resize(self.original_image, self.scale_factor)
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Update Scale", time.time() - start_time)

    def scale_up_(self, e):
        self.scale_factor += 0.1
        self.update_scaled_image_()

    def scale_down_(self, e):
        if self.scale_factor > 0.1:
            self.scale_factor -= 0.1
        self.update_scaled_image_()

    def rotate_image_(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        self.image = rotate(self.image)
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Rotate 90°", time.time() - start_time)

    def invert_colors_(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        self.image = bitwise_not(self.image)
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Invert Colors", time.time() - start_time)

    def grayscale_image_(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        # grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grayscale_bgr = grayscale(self.image)
        self.image = grayscale_bgr
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Grayscale", time.time() - start_time)

    # project 2 + opencv
    # def bilinear_zoom_in(self, e):
    #     check_if_image_exist(self.image)
    #     return False

    def bilinear_shear(self, e):
        start_time = time.time()
        
        # https://stackoverflow.com/questions/57881430/how-could-i-implement-a-centered-shear-an-image-with-opencv

        # image size
        h,w = self.original_image.shape[:2]
        # tan   
        tan = math.tan(math.radians(self.shear_angle))
        # affine transform matrix_1
        shear_matrix = np.array([
            [1, tan, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        # origin coordinate
        origin_coords = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]])
        
        # shear coordinate 
        shear_coords = (shear_matrix @ origin_coords.T).T.astype(int)



        # shear image with opencv
        self.image = cv2.warpAffine(self.original_image, shear_matrix[:2], 
                                        dsize=(np.max(shear_coords[:, 0]),  # height
                                        np.max(shear_coords[:, 1])),  # width
                                        borderValue = (0, 0, 0)
                                    )

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        # time comsuming
        self.add_time_record("Shear_1 from Bilinear", time.time() - start_time)
        return False
    
    def bilinear_perspective(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        # original position
        src_points = np.float32([
            [100, 100],  # left-up
            [400, 100],  # right-up
            [100, 400],  # left-up
            [400, 400]   # left-down
        ])

        # target position
        dst_points = np.float32([
            [50, 50],    # left-up
            [450, 50],   # right-up
            [100, 400],  # left-up
            [400, 400]   # left-down
        ])
        # get perspective matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # apply perspective transform
        h, w = self.image.shape[:2]
        self.image = cv2.warpPerspective(self.image, perspective_matrix, (w, h))

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Perspective from Bilinear", time.time() - start_time)
        return False
    
    def histogram_equalization(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        
        # convert image to grayscale image
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # grayscale_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # histogram equalization
        self.image = cv2.equalizeHist(grayscale)

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Histogram Equalization", time.time() - start_time)
        
    def histogram_plot(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        # convert image to grayscale image
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grayscale_bgr = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # # get histogram of grayscale image
        # https://www.geeksforgeeks.org/python-opencv-cv2-calchist-method/
        hist = cv2.calcHist([grayscale_bgr], [0], None, [256], [0,256])

        # size of output image
        h,w = self.image.shape[:2]
        # h,w=480, 640
        # print('h,w', h, w)
        hist_image = np.zeros((h,w,3),dtype=np.uint8)

        # normalize histogram
        cv2.normalize(hist, hist, alpha=0, beta=h, norm_type=cv2.NORM_MINMAX)

        # width of each bin
        bin_width = int(w/256)
        for i in range(256):
            # height of each bin
            bin_height = int(hist[i])
            # histogram of each rectangle
            cv2.rectangle(hist_image,
                        (i * bin_width, h - bin_height),
                        ((i + 1) * bin_width, h),
                        (144, 238, 144),  # 白色
                        -1)  # 填充矩形
        self.image = hist_image
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Histogram Plot", time.time() - start_time)

    def add_gaussian(self,e):
        check_if_image_exist(self.image)
        start_time = time.time()
        # float 32
        image = np.array(self.original_image, dtype=np.float32)
        
        # generate gaussian noise, mean=0,std=25
        noise = np.random.normal(0, 25, image.shape)

        # add noise and clip to [0,255]
        noisy_image = np.clip(image + noise, 0, 255)

        # return to uint8
        noisy_image.astype(np.uint8)
        self.image = noisy_image

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Add Gaussian", time.time() - start_time)
        
    def add_salt_pepper(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        # make sure the image is numpy
        noisy_image = self.image.copy()

        # generate random number mask, salt_prob=0.01, pepper_prob=0.01
        total_pixels = self.image.size
        salt_num = int(total_pixels * 0.01)
        pepper_num = int(total_pixels * 0.01)

        # salt
        coords_salt = [np.random.randint(0, i - 1, salt_num) for i in self.image.shape[:2]]
        noisy_image[coords_salt[0], coords_salt[1]] = 255

        # pepper
        coords_pepper = [np.random.randint(0, i - 1, pepper_num) for i in self.image.shape[:2]]
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0

        self.image = noisy_image
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Add Salt", time.time() - start_time)
        
    def median(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        kernel_size = 3
        # check  kernel size 
        if kernel_size %2 == 0:
            raise ValueError("kernel_size must be an odd number.")
        
        # deblur
        # https://stackoverflow.com/questions/78459644/medianblur-gives-error-when-i-put-it-in-another-function
        self.image=cv2.medianBlur(np.uint8(self.image), kernel_size)

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Median", time.time() - start_time)
    
    def sharpen_robert(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        # gradient
        grad_x = cv2.filter2D(self.image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(self.image, cv2.CV_64F, kernel_y)

        # convert to uint8
        grad = cv2.magnitude(grad_x, grad_y)
        grad_image = np.uint8(np.clip(grad, 0, 255))
        sharpened_image = cv2.addWeighted(self.image, 1, grad_image, 1, 0)

        # make sure image in [0,255]
        self.image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Sharpen Robert Operator", time.time() - start_time)
    
    def sharpen_sobel(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        grad_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)

        # convert to uint8
        grad = cv2.magnitude(grad_x, grad_y)
        grad_image = np.uint8(np.clip(grad, 0, 255))

        # grayscale
        # gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # print('grad_image',grad_image.shape)
        # print('gray_image',gray_image.shape)

        # origin image + edge
        sharpened_image = cv2.addWeighted(self.image, 1, grad_image, 1, 0)

        # make sure image in [0,255]
        self.image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Sharpen Sobel Operator", time.time() - start_time)
    
    def sharpen_laplace(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        # gradient (oder 2)
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F, ksize=3)
        grad_image = np.uint8(np.clip(laplacian, 0, 255))

        sharpened_image = cv2.addWeighted(self.image, 1, grad_image, 1, 0)

        # convert to uint8
        self.image = np.uint8(np.clip(np.abs(sharpened_image), 0, 255))
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Sharpen Laplace Operator", time.time() - start_time)

    def frequency_pass_filter(self, highpass=False, filter_type='ideal'): 
        d0 = 50 # Cutoff frequency
        n = 1
        # convert to grayscale image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # get size of image
        rows, cols = self.image.shape
        crow, ccol = rows // 2, cols // 2
        # calculate FFT to frequency domain
        dft = np.fft.fft2(self.image)
 
        # move to center
        dft_shift = np.fft.fftshift(dft)

        # mask or filter
        u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)  # distance

        # create low-filter mask
        if filter_type == 'ideal':
            mask = (d <= d0).astype(np.float64)
        elif filter_type == 'butterworth':
            n = 2
            mask = 1 / (1 + (d / d0) ** (2 * n))
        elif filter_type == 'gaussian':
            mask = np.exp(-(d ** 2) / (2 * (d0 ** 2)))

        # if this is high-pass-filter
        if highpass:
            mask = 1 - mask

        # apply mask 
        dft_filtered = dft_shift * mask

        # transform to space domain
        dft_ishift = np.fft.ifftshift(dft_filtered)
        img_filtered = np.fft.ifft2(dft_ishift)

        return np.abs(img_filtered)


    def frequency_low_pass_ideal(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        self.image = self.frequency_pass_filter()

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Low-pass-Ideal", time.time() - start_time)


    def frequency_high_pass_ideal(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        self.image = self.frequency_pass_filter(highpass=True)
        
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("High-pass-Ideal", time.time() - start_time)


    def frequency_low_pass_butterworth(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        self.image = self.frequency_pass_filter(filter_type='butterworth')


        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Low-pass-Butterworth", time.time() - start_time)


    def frequency_high_pass_butterworth(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        
        self.image = self.frequency_pass_filter(highpass=True, filter_type='butterworth')

        
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("High-pass-Butterworth", time.time() - start_time)


    def frequency_low_pass_gaussian(self, e):
        check_if_image_exist(self.image)

        start_time = time.time()
        self.image = self.frequency_pass_filter(filter_type='gaussian')
        
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Low-pass-Gaussian", time.time() - start_time)


    def frequency_high_pass_gaussian(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        
        self.image = self.frequency_pass_filter(highpass=True, filter_type='gaussian')

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("High-pass-Gaussian", time.time() - start_time)


    # project 2 + reimplement
    # def bilinear_zoom_in_m(self, e):
    #     check_if_image_exist(self.image)
    #     start_time = time.time()

    #     self.scale_up_(e)

    #     # update page
    #     self.image_src.src_base64 = self.to_base64(self.image)
    #     self.image_src.update()
    #     self.add_time_record("Bilinear Zoom in", time.time() - start_time)

    def bilinear_shear_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        self.image = warpAffine_m(self.original_image, self.shear_angle)

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Bilinear Shear Manually", time.time() - start_time)
    
    def shear_up_m(self, e):
        check_if_image_exist(self.image)
        if self.shear_angle < 90:
            self.shear_angle += 10
        self.bilinear_shear_m(e)


    def shear_down_m(self, e):
        check_if_image_exist(self.image)
        if self.shear_angle > -90:
            self.shear_angle -= 10
        self.bilinear_shear_m(e)

    def shear_1_up(self, e):
        check_if_image_exist(self.image)
        if self.shear_angle < 90:
            self.shear_angle += 10
        self.bilinear_shear(e)


    def shear_1_down(self, e):
        check_if_image_exist(self.image)
        if self.shear_angle > -90:
            self.shear_angle -= 10
        self.bilinear_shear(e)

    def bilinear_perspective_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        self.image = perspective_m(self.image)
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Bilinear Perspective Manually", time.time() - start_time)
    
    def histogram_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        # turn image to grayscale
        # image_rgb = np.array(self.image)
        # assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3

        # # extract rgb
        # R, G, B = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]

        # # weighted average to get the garyscale image = cv2.COLOR_BGR2GRAY
        # gray_image = 0.299 * R + 0.587 * G + 0.114 * B
        gray_image = grayscale_single_channel(self.image)
        # equalization
        self.image = equalization(gray_image)

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Histogram Equalization  Manually", time.time() - start_time)
    
    def median_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        # median function
        self.image = median_filter(self.image)

        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Median Manually", time.time() - start_time)
    
    def sharpen_robert_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()

        # convert to grayscale image
        # image = grayscale_single_channel(self.image)
        # self.image = robert_opt(self.image)
        self.image = robert_opt(self.image)
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Sharpen Robert  Manually", time.time() - start_time)
    
    def sharpen_sobel_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        
        # self.image = grayscale_single_channel(self.image)
        # self.image = sobel_opt(self.image)
        self.image = sobel_opt(self.image)
        
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Sharpen Sobel Manually", time.time() - start_time)
    
    def sharpen_laplace_m(self, e):
        check_if_image_exist(self.image)
        start_time = time.time()
        
        # self.image = grayscale_single_channel(self.image)
        self.image = laplace_opt(self.image)
        
        # update page
        self.image_src.src_base64 = self.to_base64(self.image)
        self.image_src.update()
        self.add_time_record("Sharpen Laplace Manually", time.time() - start_time)
    
    
    


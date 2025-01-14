#Adapted from: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/master/interactive_demo/app.py
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import cv2
import numpy as np
from PIL import Image
import colorsys
from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame
from dynamite.utils.misc import color_map

class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, cfg, model):
        super().__init__(master)

        # init UI window
        self.master = master
        master.title("DynaMITe: Dynamic Query Bootstrapping for Multi-object Interactive Segmentation Transformer")
        master.withdraw()
        master.update_idletasks()
        # center the window
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        #master.geometry("+%d+%d" % (x, y))
        master.geometry("+200+400")
        self.pack(fill="both", expand=True)
        
        self._input_file =None              # stores input file name
        self.show_masks_only = False
        self.is_bg_click = False

        # model
        self.controller = InteractiveController(model, update_image_callback=self._update_image, cfg=cfg)
        
        self.xPos = 0
        self.num_instances = 0
        self.prev_pressed = None
        self.buttons = {}
        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

    def _init_state(self):
        self.state = {
            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save masks with clicks', command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.save_mask_only_btn = FocusButton(self.menubar, text='Save masks only', command=self._save_mask_only_callback)
        self.save_mask_only_btn.pack(side=tk.LEFT)
        self.save_mask_only_btn.configure(state=tk.DISABLED)

        self.save_binary_mask_btn = FocusButton(self.menubar, text='Save binary masks', command=self._save_binary_mask_callback)
        self.save_binary_mask_btn.pack(side=tk.LEFT)
        self.save_binary_mask_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Save click map', command=self._save_point_clicks_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        
        self.ADD = FocusButton(self.clicks_options_frame, text="add instance", fg="green",
                    command=self._add_button)
       
        self.ADD.grid(row=0, column=1, padx=10, pady=3, sticky='w')

        self.REMOVE = FocusButton(self.clicks_options_frame, text="show masks only", fg="green",
                    command=self._show_masks_only)
       
        self.REMOVE.grid(row=0, column=2, padx=10, pady=3, sticky='w')

        self.BG_CLICK = FocusButton(self.clicks_options_frame, text="bg clicks", fg="green",
                    command=self.bg_click)
       
        self.BG_CLICK.grid(row=0, column=3, padx=10, pady=3, sticky='w')
        self.orig_bg_color = self.BG_CLICK.cget("background")

        self.resetting_clicks_frame = FocusLabelFrame(master, text="Resetting clicks and masks")
        self.resetting_clicks_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        self.reset_clicks = FocusButton(self.resetting_clicks_frame, text="reset clicks", fg="green",
                    command=self._reset_clicks)
       
        self.reset_clicks.grid(row=0, column=0, padx=10, pady=3, sticky='w')


        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=10, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):

            # browse to the image file
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Chose an image")
            self._input_file = filename
            if len(filename) > 0:
                for key in self.buttons:
                    self.buttons[key].destroy()
                
                # initialize more properties
                self.num_instances = 0
                self.prev_pressed = None
                self.is_bg_click = False
                self.BG_CLICK.configure(fg= 'green')
                self.xPos = 0
                
                # reads image as numpy array
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                self.controller.set_image(image)    
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)
                self.save_mask_only_btn.configure(state=tk.NORMAL)
                self.save_binary_mask_btn.configure(state=tk.NORMAL)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        image, _ = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),reset_clicks=False)

        filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
            ("PNG image", "*.png"),
            ("BMP image", "*.bmp"),
            ("All files", "*.*"),
        ], title="Save the current mask as...")

        if len(filename) > 0:
            im = cv2.cvtColor(image,  cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (self.controller._inputs["width"],self.controller._inputs["height"]))
            cv2.imwrite(filename, im)
    
    def _save_mask_only_callback(self):
        self.menubar.focus_set()
        image, _ = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),
                                                  reset_clicks=False, show_only_masks=True)

        filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
            ("PNG image", "*.png"),
            ("BMP image", "*.bmp"),
            ("All files", "*.*"),
        ], title="Save the current mask as...")

        if len(filename) > 0:
            im = cv2.cvtColor(image,  cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (self.controller._inputs["width"],self.controller._inputs["height"]))
            cv2.imwrite(filename, im)
    
    def _save_binary_mask_callback(self):
        self.menubar.focus_set()
        masks = self.controller.result_masks
        if masks is None:
            return
        filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save the current mask as...")
        if len(filename)>0:
            masks = np.asarray(masks.to('cpu'),dtype=np.uint8)
            comb_mask = np.zeros(masks[0].shape, dtype=np.uint8)
            for m in masks:
                comb_mask = np.logical_or(comb_mask,m)
            comb_mask = comb_mask.astype(np.uint8)
            comb_mask = cv2.resize(comb_mask,(self.controller._inputs["width"],self.controller._inputs["height"]))
            cv2.imwrite(filename, comb_mask.astype(np.uint8)*255)

    def _save_point_clicks_callback(self):
        self.menubar.focus_set()
        _, point_clicks_map = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),reset_clicks=False)

        filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
            ("PNG image", "*.png"),
            ("BMP image", "*.bmp"),
            ("All files", "*.*"),
        ], title="Save the current mask as...")

        if len(filename) > 0:
            im = cv2.cvtColor(point_clicks_map,  cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, im)

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Modified from:",
            "K.Sofiiuk and I. Petrov, (RITM Paper)",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _show_masks_only(self):
       
        if not self.show_masks_only:
            self.REMOVE.configure(fg= 'green', bg = 'red')
        else:
            self.REMOVE.configure(bg=self.orig_bg_color)
        self.show_masks_only = not self.show_masks_only
        self._update_image(show_only_masks=self.show_masks_only)

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _click_callback(self, is_positive, x, y):

        if (self.prev_pressed is None) and not self.is_bg_click:
            print("select an instance")
            return
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            n = 1 if self.is_bg_click else self.prev_pressed
            self.controller.add_click(x, y, bg_click = self.is_bg_click, inst_num=n+1)

    def _update_image(self, reset_canvas=False, reset_clicks=False, show_only_masks=False):
        image, _ = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),reset_clicks=reset_clicks,
                                                  show_only_masks=show_only_masks)
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        # self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _reset_clicks(self):

        for key in self.buttons:
            self.buttons[key].destroy()
        self.num_instances = 0
        self.prev_pressed = None
        if self.is_bg_click:
            self.is_bg_click = False
            self.BG_CLICK.configure(bg= self.orig_bg_color)
        
        self.xPos = 0
        self.controller.reset_clicks()

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked

    def _add_button(self):
        self.xPos+=1
        color_c = get_color_from_map(self.num_instances)
        self.buttons[self.num_instances] = FocusButton(self.clicks_options_frame, width=5, bg = color_c,
                                            command = lambda f=self.num_instances: self.pressed(f))
        self.buttons[self.num_instances].grid(row=self.xPos, column=1, padx=10, pady=2, sticky='E')
        self.pressed(self.num_instances)
        print(f"Added button for instance:{self.num_instances}")
        self.num_instances+=1
    
    def remove_button(self):
        if self.prev_pressed == self.num_instances-1:
             self.prev_pressed = None
        if self.num_instances< 1:
            print("No more instances to remove")
            return
        self.buttons[self.num_instances-1].destroy()
        print(f'Removed instance: {self.num_instances-1}')
        # self.buttons[self.num_instances-1].pack()
        self.num_instances-=1

    def pressed(self, index):
        print("Selected instance:", index)
        if self.prev_pressed is not None:
            color_c = get_color_from_map(self.prev_pressed)
            self.buttons[self.prev_pressed].configure(bg =color_c, width=5, text="")
        if self.is_bg_click:
            self.is_bg_click = False
            self.BG_CLICK.configure(bg= self.orig_bg_color)
        self.prev_pressed = index
        
        c = change_color_brightness(color_map[index],2.2)
        color_c='#%02x%02x%02x' % (c[0],c[1], c[2])
        # color_c='#%02x%02x%02x' % (color_map[2*(index)+2][0], color_map[2*(index)+2][1], color_map[2*(index)+2][2])
        self.buttons[index].configure(bg = color_c, width=5, text = f"selected")
        
    def bg_click(self):
        if self.prev_pressed is not None:
            color_c = get_color_from_map(self.prev_pressed)
            self.buttons[self.prev_pressed].configure(bg = color_c, width=5, text="")
            self.prev_pressed = None
        self.is_bg_click = True
        self.BG_CLICK.configure(fg= 'green', bg = 'red')

def get_color_from_map(index):
    color_c='#%02x%02x%02x' % (color_map[index][0], color_map[index][1], color_map[index][2])
    return color_c

def change_color_brightness(color, brightness_factor):
       
    r, g, b = [x/255.0 for x in color]
    polygon_color =  colorsys.rgb_to_hls(r,g,b)
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])

    return np.asarray([c*255 for c in modified_color], dtype =np.uint8)



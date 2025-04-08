import cv2
import numpy as np
import threading
import os
import sys

# Global variables
selected_color = (255, 165, 0)  # Default orange color in BGR
lower_hsv = np.array([10, 100, 70])
upper_hsv = np.array([25, 255, 255])
range_val = 20
eyedropper_active = False
mouse_x, mouse_y = 0, 0
is_running = True

# Check if tkinter is available
has_tkinter = False
try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    has_tkinter = True
except ImportError:
    print("Warning: tkinter not available. Running in command-line mode only.")
    print("To install tkinter, try:")
    if sys.platform.startswith('linux'):
        print("  sudo apt-get install python3-tk")
    elif sys.platform.startswith('darwin'):
        print("  brew install python-tk")
    elif sys.platform.startswith('win'):
        print("  Make sure to install Python with tcl/tk support")

# Initialize tkinter-related variables
color_panel_img = None
tk_root = None
slider_frame = None
color_panel = None
h_min_slider = None
h_max_slider = None
s_min_slider = None
s_max_slider = None
v_min_slider = None
v_max_slider = None
range_slider = None
r_slider = None
g_slider = None
b_slider = None

def update_color_from_rgb(r, g, b, range_value):
    """
    Updates HSV range based on RGB color and range value
    """
    global lower_hsv, upper_hsv, range_val
    
    # Store range value
    range_val = range_value
    
    # Convert RGB to HSV
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Set HSV range
    lower_hsv = np.array([max(0, int(h) - range_val), max(0, int(s) - range_val), max(0, int(v) - range_val)])
    upper_hsv = np.array([min(179, int(h) + range_val), min(255, int(s) + range_val), min(255, int(v) + range_val)])
    
    # Update sliders if they exist
    if has_tkinter and slider_frame:
        update_sliders()

def rgb_to_hsv(r, g, b):
    """
    Convert RGB values to HSV
    
    Args:
        r, g, b: RGB values (0-255)
        
    Returns:
        h, s, v: HSV values
    """
    # Convert to 0-1 range for OpenCV
    r, g, b = r/255.0, g/255.0, b/255.0
    
    # Convert using OpenCV
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    
    return hsv[0], hsv[1], hsv[2]

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for eyedropper tool
    """
    global eyedropper_active, mouse_x, mouse_y, selected_color
    
    # Store mouse position
    mouse_x, mouse_y = x, y
    
    # On left click, sample the color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            # Get color at clicked position (BGR)
            b, g, r = frame[y, x]
            selected_color = (b, g, r)
            
            # Update HSV range based on this color
            update_color_from_rgb(r, g, b, range_val)
            
            # Update color panel
            if has_tkinter:
                update_color_panel()

def color_panel_click(event):
    """
    Handle clicks on the color panel
    """
    global selected_color
    
    if has_tkinter and color_panel_img is not None:
        # Get color at clicked position
        r, g, b = color_panel_img.getpixel((event.x, event.y))
        selected_color = (b, g, r)  # Convert to BGR for OpenCV
        
        # Update HSV range based on this color
        update_color_from_rgb(r, g, b, range_val)
        
        # Update the color panel
        update_color_panel()

def update_sliders():
    """
    Update the slider values based on current HSV range
    """
    if not has_tkinter or not slider_frame:
        return
        
    # Update HSV min sliders
    h_min_slider.set(lower_hsv[0])
    s_min_slider.set(lower_hsv[1])
    v_min_slider.set(lower_hsv[2])
    
    # Update HSV max sliders
    h_max_slider.set(upper_hsv[0])
    s_max_slider.set(upper_hsv[1])
    v_max_slider.set(upper_hsv[2])
    
    # Update range slider
    range_slider.set(range_val)
    
    # Update RGB sliders
    b, g, r = selected_color
    r_slider.set(r)
    g_slider.set(g)
    b_slider.set(b)

def update_from_sliders():
    """
    Update HSV range based on slider values
    """
    global lower_hsv, upper_hsv, range_val, selected_color
    
    if not has_tkinter:
        return
        
    # Get values from sliders
    h_min = h_min_slider.get()
    s_min = s_min_slider.get()
    v_min = v_min_slider.get()
    
    h_max = h_max_slider.get()
    s_max = s_max_slider.get()
    v_max = v_max_slider.get()
    
    range_val = range_slider.get()
    
    # Update HSV ranges
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    
    # Update RGB color
    r = r_slider.get()
    g = g_slider.get()
    b = b_slider.get()
    selected_color = (b, g, r)
    
    # Update color panel
    update_color_panel()

def update_color_panel():
    """
    Update the color panel with the current selected color
    """
    global color_panel_img
    
    if not has_tkinter or not color_panel:
        return
        
    # Create color panel image
    width, height = 300, 200
    color_panel_img = Image.new('RGB', (width, height))
    
    # Get BGR color
    b, g, r = selected_color
    
    # Fill top half with selected color
    for x in range(width):
        for y in range(height // 2):
            color_panel_img.putpixel((x, y), (r, g, b))
    
    # Create gradient for bottom half showing the range
    h, s, v = rgb_to_hsv(r, g, b)
    for x in range(width):
        # Calculate hue based on position
        hue_offset = (x / width * 2 - 1) * range_val
        hue = max(0, min(179, h + hue_offset))
        
        # Convert back to RGB for display
        color_hsv = np.uint8([[[hue, s, v]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        b_grad, g_grad, r_grad = color_bgr
        
        for y in range(height // 2, height):
            color_panel_img.putpixel((x, y), (r_grad, g_grad, b_grad))
    
    # Convert to PhotoImage and update canvas
    photo = ImageTk.PhotoImage(color_panel_img)
    color_panel.create_image(0, 0, image=photo, anchor=tk.NW)
    color_panel.image = photo  # Keep a reference to prevent garbage collection

def create_ui():
    """
    Create the Tkinter UI for better sliders and color panel
    """
    global tk_root, slider_frame, h_min_slider, h_max_slider, s_min_slider, s_max_slider
    global v_min_slider, v_max_slider, range_slider, r_slider, g_slider, b_slider, color_panel
    
    if not has_tkinter:
        print("Cannot create UI: tkinter not available")
        return
    
    # Create Tkinter root window
    tk_root = tk.Tk()
    tk_root.title("Color Picker")
    tk_root.geometry("350x700")
    tk_root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Create color panel
    color_panel_frame = ttk.LabelFrame(tk_root, text="Color Panel (Click to sample)")
    color_panel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    color_panel = tk.Canvas(color_panel_frame, width=300, height=200, bg="black")
    color_panel.pack(padx=10, pady=10)
    color_panel.bind("<Button-1>", color_panel_click)
    
    # Create slider frame
    slider_frame = ttk.LabelFrame(tk_root, text="Color Controls")
    slider_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # RGB sliders
    ttk.Label(slider_frame, text="RGB Values:").pack(anchor=tk.W, padx=10, pady=(10, 0))
    
    r_frame = ttk.Frame(slider_frame)
    r_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(r_frame, text="R:").pack(side=tk.LEFT)
    r_slider = ttk.Scale(r_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    r_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    r_slider.set(selected_color[2])  # R value
    
    g_frame = ttk.Frame(slider_frame)
    g_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(g_frame, text="G:").pack(side=tk.LEFT)
    g_slider = ttk.Scale(g_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    g_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    g_slider.set(selected_color[1])  # G value
    
    b_frame = ttk.Frame(slider_frame)
    b_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(b_frame, text="B:").pack(side=tk.LEFT)
    b_slider = ttk.Scale(b_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    b_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    b_slider.set(selected_color[0])  # B value
    
    # Range slider
    range_frame = ttk.Frame(slider_frame)
    range_frame.pack(fill=tk.X, padx=10, pady=(10, 2))
    ttk.Label(range_frame, text="Range:").pack(side=tk.LEFT)
    range_slider = ttk.Scale(range_frame, from_=0, to=50, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    range_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    range_slider.set(range_val)
    
    # HSV Min sliders
    ttk.Label(slider_frame, text="HSV Min Values:").pack(anchor=tk.W, padx=10, pady=(10, 0))
    
    h_min_frame = ttk.Frame(slider_frame)
    h_min_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(h_min_frame, text="H min:").pack(side=tk.LEFT)
    h_min_slider = ttk.Scale(h_min_frame, from_=0, to=179, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    h_min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    h_min_slider.set(lower_hsv[0])
    
    s_min_frame = ttk.Frame(slider_frame)
    s_min_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(s_min_frame, text="S min:").pack(side=tk.LEFT)
    s_min_slider = ttk.Scale(s_min_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    s_min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    s_min_slider.set(lower_hsv[1])
    
    v_min_frame = ttk.Frame(slider_frame)
    v_min_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(v_min_frame, text="V min:").pack(side=tk.LEFT)
    v_min_slider = ttk.Scale(v_min_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    v_min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    v_min_slider.set(lower_hsv[2])
    
    # HSV Max sliders
    ttk.Label(slider_frame, text="HSV Max Values:").pack(anchor=tk.W, padx=10, pady=(10, 0))
    
    h_max_frame = ttk.Frame(slider_frame)
    h_max_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(h_max_frame, text="H max:").pack(side=tk.LEFT)
    h_max_slider = ttk.Scale(h_max_frame, from_=0, to=179, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    h_max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    h_max_slider.set(upper_hsv[0])
    
    s_max_frame = ttk.Frame(slider_frame)
    s_max_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(s_max_frame, text="S max:").pack(side=tk.LEFT)
    s_max_slider = ttk.Scale(s_max_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    s_max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    s_max_slider.set(upper_hsv[1])
    
    v_max_frame = ttk.Frame(slider_frame)
    v_max_frame.pack(fill=tk.X, padx=10, pady=2)
    ttk.Label(v_max_frame, text="V max:").pack(side=tk.LEFT)
    v_max_slider = ttk.Scale(v_max_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda _: update_from_sliders())
    v_max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    v_max_slider.set(upper_hsv[2])
    
    # Instructions
    instructions = ttk.Label(tk_root, text="Click on video to sample colors\nPress 'q' to quit")
    instructions.pack(pady=10)
    
    # Update color panel
    update_color_panel()
    
    # Start Tkinter event loop in a separate thread
    threading.Thread(target=tk_root.mainloop, daemon=True).start()

def on_closing():
    """
    Handle window closing
    """
    global is_running
    is_running = False
    if has_tkinter and tk_root:
        tk_root.destroy()

def detect_color(frame, lower_hsv, upper_hsv):
    """
    Detects objects in the frame based on HSV color range and draws boxes around them.
    
    Args:
        frame: The input image frame
        lower_hsv: Lower bound of HSV range
        upper_hsv: Upper bound of HSV range
        
    Returns:
        frame: The frame with boxes drawn around detected objects
        mask: The binary mask showing detected objects
        result: The frame showing only the detected color
        centers: List of (x, y) coordinates of object centers
    """
    # Convert to HSV color space for better color detection
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for the selected color range
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Apply the mask to get only the selected color
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize list to store object centers
    centers = []
    
    # Process each contour
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 100:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle around the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate and store center
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append((center_x, center_y))
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
        
    return frame, mask, result, centers

if __name__ == "__main__":
    # Create the UI if tkinter is available
    if has_tkinter:
        create_ui()
    else:
        print("Running in command-line mode (no GUI)")
        print(f"Initial HSV range: {lower_hsv} to {upper_hsv}")
        print("You can still use the video windows and click to sample colors")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use default camera (change to 1 or other index if needed)
    
    # Set up mouse callback for eyedropper
    cv2.namedWindow("Original")
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Set mouse callback with frame as parameter
        cv2.setMouseCallback("Original", mouse_callback, frame)
        
        # Create a copy of the frame for processing
        display_frame = frame.copy()
        
        # Detect objects with the selected color and draw boxes
        processed_frame, mask, color_result, object_centers = detect_color(frame.copy(), lower_hsv, upper_hsv)
        
        # Display the number of objects detected
        cv2.putText(processed_frame, f"Objects: {len(object_centers)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display HSV values on the frame
        hsv_text = f"HSV Min: [{lower_hsv[0]}, {lower_hsv[1]}, {lower_hsv[2]}]"
        cv2.putText(processed_frame, hsv_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        hsv_text = f"HSV Max: [{upper_hsv[0]}, {upper_hsv[1]}, {upper_hsv[2]}]"
        cv2.putText(processed_frame, hsv_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display RGB values on the frame
        b, g, r = selected_color
        rgb_text = f"RGB: [{r}, {g}, {b}]"
        cv2.putText(processed_frame, rgb_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show instructions for clicking
        cv2.putText(display_frame, "Click to sample color", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show the frames
        cv2.imshow("Original", display_frame)
        cv2.moveWindow("Original", 0, 0)
        
        cv2.imshow("Color Detection", processed_frame)
        cv2.moveWindow("Color Detection", 640, 0)
        
        cv2.imshow("Mask", mask)
        cv2.moveWindow("Mask", 0, 480)
        
        cv2.imshow("Color Only", color_result)
        cv2.moveWindow("Color Only", 640, 480)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    on_closing()

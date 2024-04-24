import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread

# Import your custom functions
from img_prep import prepare_images
from codebook_gen import generate_codebook
from point_cloud_reconstruction import generate_point_cloud
from reconstruction import reconstruct_3d
from visualization import visualize_mesh


# Function to handle folder selection
def select_folder():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)


# Flag to indicate whether the process is running
process_running = False

# Function to toggle buttons and progress bar
def toggle_elements(start=True):
    global process_running
    if start:
        start_button.grid(row=6, column=0, pady=10, sticky="w")
        cancel_button.grid_forget()
        process_running = False
    else:
        start_button.grid_forget()
        cancel_button.grid(row=7, column=0, pady=10, sticky="w")
        process_running = True

# Function to cancel the image processing
def cancel_process():
    global cancel_flag, process_cancelled
    log_text.insert(tk.END, "Cancelling process, please wait...\n")
    cancel_flag = True
    process_cancelled = True

    # Show the progress bar and hide the buttons
    progress_bar.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")
    toggle_elements(True)  # Show start button

    # Update the GUI
    app.update()

# Function to start the image processing
def start_process():
    global cancel_flag, process_cancelled
    cancel_flag = False
    process_cancelled = False

    # Retrieve inputs from GUI
    src_dir = folder_entry.get()

    if not src_dir:
        log_text.insert(tk.END, "Error: Please select an image folder.\n")
        return  # Prevent processing if no path is provided

    similarity_threshold_value = threshold_entry.get()

    # Set default values if not provided
    similarity_threshold_value = float(similarity_threshold_value) if similarity_threshold_value else 0.8
    max_num_keypoints = int(max_num_keypoints_entry.get()) if max_num_keypoints_entry.get() else 2048
    cluster_count = int(cluster_count_entry.get()) if cluster_count_entry.get() else 200  # Use user input or default
    iterations = int(iterations_entry.get()) if iterations_entry.get() else 1  # Use user input or default
    focal_length = float(focal_length_entry.get()) if focal_length_entry.get() else 2382.05  # Use user input or default

    # Create cache and mesh directories if they don't exist
    cache_dir = os.path.join(src_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    mesh_dir = os.path.join(src_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # Hide the start button and show the progress bar
    toggle_elements(False)  # Hide start button, show cancel button
    progress_bar.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")

    # Define a function for image processing
    def process_images(update_progress, check_cancel):
        
        try:
            # Step 1: Image Preparation
            update_progress(20)
            log_text.insert(tk.END, "Preparing Images...\n")
            descriptors_set = prepare_images(cache_dir, src_dir, max_num_keypoints)
            log_text.insert(tk.END, f"prepare_images returned: {descriptors_set}\n")

            if not src_dir:
                log_text.insert(tk.END, "Error: Please select an image folder.\n")
                return  # Prevent processing if no path is provided

            # Step 2: Codebook Generation
            update_progress(40)
            log_text.insert(tk.END, "Generating Codebook...\n")
            codebook_result = generate_codebook(cache_dir, cluster_count, iterations)
            log_text.insert(tk.END, f"generate_codebook returned: {codebook_result}\n")

            # Check for cancellation
            if check_cancel():
                log_text.insert(tk.END, "Process cancelled.\n")
                return

            # Step 3: Point Cloud Reconstruction
            update_progress(60)
            log_text.insert(tk.END, "Reconstructing Point Cloud...\n")
            point_cloud_result = generate_point_cloud(cache_dir, similarity_threshold_value)
            log_text.insert(tk.END, f"generate_point_cloud returned: {point_cloud_result}\n")

            # Check for cancellation
            if check_cancel():
                log_text.insert(tk.END, "Process cancelled.\n")
                return

            # Step 4: 3D Reconstruction
            update_progress(80)
            log_text.insert(tk.END, "Performing 3D Reconstruction...\n")
            reconstruct_3d_result = reconstruct_3d(cache_dir, mesh_dir, focal_length)
            log_text.insert(tk.END, f"reconstruct_3d returned: {reconstruct_3d_result}\n")

            # Check for cancellation
            if check_cancel():
                log_text.insert(tk.END, "Process cancelled.\n")
                return

            # Step 5: Visualization
            update_progress(100)
            log_text.insert(tk.END, "Visualizing 3D Mesh...\n")
            visualize_mesh_result = visualize_mesh(mesh_dir)
            log_text.insert(tk.END, f"visualize_mesh returned: {visualize_mesh_result}\n")


            # Show completion message
            log_text.insert(tk.END, "Process completed.\n")

        except Exception as e:
            log_text.insert(tk.END, f"Error: {str(e)}\n")

        finally:
            # After processing, update GUI elements
            progress_bar.grid_remove()
            if not process_cancelled:
                toggle_elements(True)  # Show start button

            app.update()  # Force update to show the log messages

            


    # Define functions for updating progress and checking cancel flag
    def update_progress(value):
        progress_bar["value"] = value
        app.update()

    def check_cancel():
        return cancel_flag

    # Start a thread for image processing
    processing_thread = Thread(target=process_images, args=(update_progress, check_cancel))
    processing_thread.start()

# Create the main application window
app = tk.Tk()
app.title("3D Reconstruction")

# Create a frame for better organization
frame = tk.Frame(app, padx=10, pady=10)
frame.pack()

# Label and Entry for selecting the image folder
folder_label = tk.Label(frame, text="Select Image Folder:")
folder_label.grid(row=0, column=0, sticky="w")

folder_entry = tk.Entry(frame, width=40)
folder_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

select_button = tk.Button(frame, text="Browse", command=select_folder)
select_button.grid(row=0, column=3, padx=5, pady=5)

# Function to validate float input
def validate_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Entry for setting the threshold value
threshold_label = tk.Label(frame, text="Threshold Value:")
threshold_label.grid(row=1, column=0, sticky="w")

validate_float_cmd = frame.register(validate_float)
threshold_entry = tk.Entry(frame, width=10, validate="key", validatecommand=(validate_float_cmd, '%P'))
threshold_entry.grid(row=1, column=1, padx=5, pady=5)
threshold_entry.insert(0, "0.8")  # Set default value

# Entry for setting the max number of keypoints
max_num_keypoints_label = tk.Label(frame, text="Max Num Keypoints:")
max_num_keypoints_label.grid(row=1, column=2, sticky="w")

max_num_keypoints_entry = tk.Entry(frame, width=10, validate="key", validatecommand=(validate_float_cmd, '%P'))
max_num_keypoints_entry.grid(row=1, column=3, padx=5, pady=5)
max_num_keypoints_entry.insert(0, "2048")  # Set default value

# Label and Entry for cluster count
cluster_count_label = tk.Label(frame, text="Cluster Count:")
cluster_count_label.grid(row=2, column=0, sticky="w")

cluster_count_entry = tk.Entry(frame, width=10, validate="key", validatecommand=(validate_float_cmd, '%P'))
cluster_count_entry.grid(row=2, column=1, padx=5, pady=5)
cluster_count_entry.insert(0, "200")  # Set default value

# Label and Entry for iterations
iterations_label = tk.Label(frame, text="Iterations:")
iterations_label.grid(row=2, column=2, sticky="w")

iterations_entry = tk.Entry(frame, width=10, validate="key", validatecommand=(validate_float_cmd, '%P'))
iterations_entry.grid(row=2, column=3, padx=5, pady=5)
iterations_entry.insert(0, "1")  # Set default value

# Label and Entry for focal length
focal_length_label = tk.Label(frame, text="Focal Length:")
focal_length_label.grid(row=3, column=0, sticky="w")

focal_length_entry = tk.Entry(frame, width=10, validate="key", validatecommand=(validate_float_cmd, '%P'))
focal_length_entry.grid(row=3, column=1, padx=5, pady=5)
focal_length_entry.insert(0, "2382.05")  # Set default value

# Create a Text widget for displaying logs
log_text = tk.Text(frame, width=40, height=10)
log_text.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

# Create a scrollbar for the Text widget
log_scrollbar = tk.Scrollbar(frame, command=log_text.yview)
log_scrollbar.grid(row=4, column=4, sticky="ns")
log_text.config(yscrollcommand=log_scrollbar.set)

# Create a Progressbar widget
progress_bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
progress_bar.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")

# Button to start the image processing
start_button = tk.Button(frame, text="Start Process", command=start_process)
start_button.grid(row=6, column=0, pady=10, sticky="w")

# Button to cancel the image processing
cancel_button = tk.Button(frame, text="Cancel Process", command=cancel_process)

# Start the tkinter main loop
app.mainloop()

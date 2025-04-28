import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import os
import sys

# Function to load the model and scaler (adjusted for PyInstaller)
def load_model():
    try:
        # Determine base path for bundled files (use _MEIPASS for bundled exe)
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS  # PyInstaller extraction folder
        else:
            base_path = os.path.dirname(__file__)  # Current directory when running as a script

        # Paths to the .pkl files
        model_path = os.path.join(base_path, 'antenna_model.pkl')
        scaler_path = os.path.join(base_path, 'scaler.pkl')

        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model and scaler: {e}")
        return None, None

# Load the model and scaler
model, scaler = load_model()

# Define targets for display
targets = ['Gain_dB', 'Directivity', 'S11_dB', 'Bandwidth_MHz', 'Radiation_Efficiency']

# Define the prediction function
def predict_antenna():
    try:
        # Get values from input fields
        design_parameters = [
            float(entry_patch_length.get()),
            float(entry_patch_width.get()),
            float(entry_slot_length.get()),
            float(entry_slot_width.get()),
            float(entry_substrate_height.get()),
            float(entry_relative_permittivity.get()),
            float(entry_frequency.get())
        ]
        
        # Preprocess and predict
        design_parameters = np.array(design_parameters).reshape(1, -1)
        scaled_parameters = scaler.transform(design_parameters)
        predictions = model.predict(scaled_parameters)[0]
        
        # Display predictions
        result_text = "\n".join([f"{target}: {pred:.2f}" for target, pred in zip(targets, predictions)])
        result_label.config(text=result_text)
    
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or prediction error:\n{e}")

# Create the GUI
root = tk.Tk()
root.title("Antenna Performance Predictor")

# Input fields
tk.Label(root, text="Patch Length (mm):").grid(row=0, column=0, padx=10, pady=5)
entry_patch_length = tk.Entry(root)
entry_patch_length.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Patch Width (mm):").grid(row=1, column=0, padx=10, pady=5)
entry_patch_width = tk.Entry(root)
entry_patch_width.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Slot Length (mm):").grid(row=2, column=0, padx=10, pady=5)
entry_slot_length = tk.Entry(root)
entry_slot_length.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Slot Width (mm):").grid(row=3, column=0, padx=10, pady=5)
entry_slot_width = tk.Entry(root)
entry_slot_width.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Substrate Height (mm):").grid(row=4, column=0, padx=10, pady=5)
entry_substrate_height = tk.Entry(root)
entry_substrate_height.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Relative Permittivity:").grid(row=5, column=0, padx=10, pady=5)
entry_relative_permittivity = tk.Entry(root)
entry_relative_permittivity.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Frequency (GHz):").grid(row=6, column=0, padx=10, pady=5)
entry_frequency = tk.Entry(root)
entry_frequency.grid(row=6, column=1, padx=10, pady=5)

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_antenna)
predict_button.grid(row=7, column=0, columnspan=2, pady=10)

# Result display
result_label = tk.Label(root, text="Predicted Performance Metrics will appear here", justify="left", fg="blue")
result_label.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

# Run the GUI loop
root.mainloop()

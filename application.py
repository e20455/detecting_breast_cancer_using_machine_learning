import tkinter as tk 
from tkinter import ttk, messagebox
import json
import bcrypt
import pandas as pd
import pickle

USERS_FILE = "users.json"

# load users from users.json
def load_users():
    try:
        with open(USERS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("[INFO] users.json not found, creating a new one.")
        return {}
    except json.JSONDecodeError:
        print("[ERROR] users.json is corrupted.")
        return {}

# save users to users.json
def save_users(users):
    with open(USERS_FILE, "w") as file:
        json.dump(users, file, indent=4)

USERS = load_users()
logged_in = False
current_user = None
feature_values = {}
submitted_features_list = []

# register new user
def register():
    username = reg_username_entry.get().strip()
    password = reg_password_entry.get().strip()
    
    if not username or not password:
        messagebox.showerror("Registration Failed", "Please enter a username and password")
        return
    
    if username in USERS:
        messagebox.showerror("Registration Failed", "Username already exists")
    else:
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        USERS[username] = hashed_password
        save_users(USERS)
        messagebox.showinfo("Registration Success", "Account created successfully")
        reg_username_entry.delete(0, tk.END)
        reg_password_entry.delete(0, tk.END)
        register_frame.pack_forget()
        login_frame.pack()

# login existing user
def login():
    global logged_in, current_user
    username = username_entry.get().strip()
    password = password_entry.get().strip()
    
    if not username or not password:
        messagebox.showerror("Login Failed", "Please enter a username and password")
        return
    
    stored_password = USERS.get(username)
    if stored_password and bcrypt.checkpw(password.encode(), stored_password.encode()):
        logged_in = True
        current_user = username
        messagebox.showinfo("Login Success", "Welcome")
        username_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)
        login_frame.pack_forget()
        menu_frame.pack(expand=True)
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# validate feature entery - must be a number and lower than 50
def validate_features(entries):
    features = {}
    for name, entry in entries.items():
        try:
            value = float(entry.get())
            if value > 50:
                messagebox.showerror("Value Too High", f"'{name}' must be 50 or less.")
                return None
            features[name] = value
        except ValueError:
            messagebox.showerror("Invalid Input", f"'{name}' must be a number.")
            return None
    return features

# on submit button analysis features and make prediction
def feature_analysis():
    global user_df, rescaled_user_df
    
    submit_features = validate_features(feature_values)
    if submit_features is None:
        return  
    
    submit_button.grid_forget()
    enter_button.grid(column=0, row=(len(feature_defaults) // 2) + 1, columnspan=4, pady=10)
    
    submit_features = {}  
    for feature, entry in feature_values.items():
        submit_features[feature] = float(entry.get())
        
    submitted_features_list.append(submit_features)
    
    model_path = "lr_model.pkl"
    try:
        with open(model_path, 'rb') as file:
            lr_model = pickle.load(file)
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Could not load the LR model:\n{e}")
        return
    
    user_df = pd.DataFrame([submit_features])
    
    scaler_path = "scaler.pkl"
    try:
        with open(scaler_path, 'rb') as file:
             scaler = pickle.load(file)
    except Exception as e:
        messagebox.showerror("Scaler Load Error", f"Could not load the scaler:\n{e}")
        return
        
    rescaled_user_df = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)
    prediction = lr_model.predict(rescaled_user_df)
    predicted_class = prediction[0]
    
    if predicted_class == 0:
        #print("Benign")
        benign_result.grid(column=0, row=(len(feature_defaults) // 2) + 2, columnspan=4, pady=10)
        malignant_result.grid_forget()
        
    else: # predicted_class == 1:
        #print("Maligant")
        malignant_result.grid(column=0, row=(len(feature_defaults) // 2) + 2, columnspan=4, pady=10)
        benign_result.grid_forget()
        
    for entry in feature_values.values():
        entry.config(state=tk.DISABLED)
    
    if 'history' not in USERS:
        USERS['history'] = {}

    if current_user not in USERS['history']:
        USERS['history'][current_user] = []

    USERS['history'][current_user].append({
        "input": submit_features,
        "prediction": int(predicted_class)
    })

    save_users(USERS)

# resets user entry fields after they are used
def reset_features():
    for entry in feature_values.values():
       entry.config(state=tk.NORMAL)
    for feature, entry in feature_values.items():
       entry.delete(0, tk.END)
       entry.insert(0, feature_defaults[feature])
    
# layout and content for view history page
def view_history():
    menu_frame.pack_forget()

    history_frame = ttk.Frame(root)
    history_frame.pack(fill="both", expand=True)

    # set up container for scrollbar
    container = ttk.Frame(history_frame)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    scrollable_frame = ttk.Frame(canvas)
    canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # resize scroll bar
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(canvas_frame, width=canvas.winfo_width())

    scrollable_frame.bind("<Configure>", on_configure)

    centered_content = ttk.Frame(scrollable_frame)
    centered_content.pack(anchor="center", pady=10)

    for i in range(4):
        centered_content.columnconfigure(i, weight=1, minsize=150)

    # get and display users history
    username = current_user
    history = USERS.get("history", {}).get(username, [])
    if not history:
        messagebox.showinfo("History", "No history found for this user.")
        history_frame.pack_forget()
        menu_frame.pack(expand=True)
        return
    
    row_offset = 0
    for idx, record in enumerate(history):
        ttk.Label(
            centered_content,
            text=f"Entry {idx + 1}: prediction is likely to be {'malignant' if record['prediction'] == 1 else 'benign'}",
            font=("Helvetica", 10, "bold")).grid(column=0, row=row_offset, columnspan=4, sticky="w", padx=5, pady=4)

        row_offset += 1
        input_items = list(record["input"].items())
        for i, (feature, value) in enumerate(input_items):
            col = i % 2
            row = i // 2
            ttk.Label(centered_content, text=f"{feature}: {value}").grid(column=col * 2, row=row_offset + row, columnspan=2, sticky="ew", padx=5, pady=2)

        row_offset += (len(input_items) + 1) // 2

    # update scrollbar
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    # main menu button 
    bottom_frame = ttk.Frame(history_frame)
    bottom_frame.pack(fill="x", pady=5)

    menu_button = ttk.Button(bottom_frame, text="Main Menu", command=lambda: [history_frame.pack_forget(), menu_frame.pack(expand=True)])
    menu_button.pack(pady=5)

# main window
root = tk.Tk()
root.title("Feature Analysis Application")
root.geometry("500x500")

# register frame
register_frame = tk.Frame(root)
ttk.Label(register_frame, text="Please register your account").pack(pady=20)
ttk.Label(register_frame, text="Username:").pack()
reg_username_entry = ttk.Entry(register_frame)
reg_username_entry.pack()
ttk.Label(register_frame, text="Password:").pack()
reg_password_entry = ttk.Entry(register_frame, show="*")
reg_password_entry.pack()
ttk.Button(register_frame, text="Create Account", command=register).pack(pady=10)
ttk.Button(register_frame, text="Back to Login", command=lambda: [register_frame.pack_forget(), login_frame.pack(expand=True)]).pack()

# login frame
login_frame = ttk.Frame(root, padding="3 3 12 12")
login_frame.pack(expand=True)
login_frame.columnconfigure(0, weight=1)
login_frame.rowconfigure(0, weight=1)
ttk.Label(login_frame, text="Please login to your account").pack(pady=20)
ttk.Label(login_frame, text="Username:").pack()
username_entry = ttk.Entry(login_frame)
username_entry.pack()
ttk.Label(login_frame, text="Password:").pack()
password_entry = ttk.Entry(login_frame, show="*")
password_entry.pack()
ttk.Button(login_frame, text='Login', command=login).pack(pady=10)
ttk.Button(login_frame, text='Register', command=lambda: [login_frame.pack_forget(), register_frame.pack(expand=True)]).pack()

# main menu frame
menu_frame = ttk.Frame(root, padding="3 3 12 12")
menu_frame.columnconfigure(0, weight=1)
menu_frame.rowconfigure(0, weight=1)
ttk.Button(menu_frame, text='Upload Features', command=lambda: [menu_frame.pack_forget(), feature_frame.pack(expand=True)]).pack(pady=10)
ttk.Button(menu_frame, text='View History', command=view_history).pack(pady=10)
ttk.Button(menu_frame, text='Log Out', command=lambda: [menu_frame.pack_forget(), login_frame.pack(expand=True)]).pack(pady=10)

# feature frame
feature_frame = ttk.Frame(root, padding="3 3 12 12")
feature_frame.columnconfigure(1, weight=1)
feature_frame.columnconfigure(3, weight=1)

feature_defaults = {
    "radius_mean": "20.57", "texture_mean": "17.77", "smoothness_mean": "0.08474", "compactness_mean": "0.07864", "symmetry_mean": "0.1812",
    "fractal_dimension_mean": "0.05667", "radius_se": "0.5435", "texture_se": "0.7339", "smoothness_se": "0.005225", "compactness_se": "0.01308",
    "concavity_se": "0.0186", "concave points_se": "0.0134", "symmetry_se": "0.01389", "fractal_dimension_se": "0.003532", "smoothness_worst": "0.1238", 
    "symmetry_worst": "0.275"
    }

feature_info = ttk.Label(feature_frame, text="Enter your feature values below:")
feature_info.grid(column=0, row=0, columnspan=4, pady=10)

for i, (feature, default_value) in enumerate(feature_defaults.items()):
    col = i % 2  
    row = i // 2  

    ttk.Label(feature_frame, text=f"{feature}:").grid(column=col * 2, row=row + 1, sticky=tk.W, padx=5, pady=2)
    entry = ttk.Entry(feature_frame)
    entry.grid(column=col * 2 + 1, row=row + 1, sticky=(tk.W, tk.E), padx=5, pady=2)
    entry.insert(0, default_value)
    feature_values[feature] = entry

submit_button = ttk.Button(feature_frame, text="Submit", command=feature_analysis)
submit_button.grid(column=0, row=(len(feature_defaults) // 2) + 1, columnspan=4, pady=10)

enter_button = ttk.Button(feature_frame, text="Enter another set of features", command=lambda: [enter_button.grid_forget(), reset_features(),
                                                                                                malignant_result.grid_forget(), benign_result.grid_forget(),
                                                                                                submit_button.grid(column=0, row=(len(feature_defaults) // 2) + 1, columnspan=4, pady=10)])
enter_button.grid(column=0, row=(len(feature_defaults) // 2) + 1, columnspan=4, pady=10)
enter_button.grid_forget()

benign_result = ttk.Label(feature_frame, text="Your sample is likely to be Benign")
malignant_result = ttk.Label(feature_frame, text="Your sample is likely to be Malignant")
benign_result.grid_forget()
malignant_result.grid_forget()

menu_button = ttk.Button(feature_frame, text="Main Menu", command=lambda: [benign_result.grid_forget(), malignant_result.grid_forget(), 
                                                                        enter_button.grid_forget(), reset_features(), feature_frame.pack_forget(),                                                                 
                                                                        submit_button.grid(column=0, row=(len(feature_defaults) // 2) + 1, columnspan=4, pady=10), menu_frame.pack(expand=True)])
menu_button.grid(column=0, row=(len(feature_defaults) // 2) + 3, columnspan=4, pady=10)


root.mainloop()
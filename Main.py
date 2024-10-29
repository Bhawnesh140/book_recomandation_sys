import os
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext, ttk
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# Set environment variables to suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load dataset
data_path = 'D:/Placement/Project/Book/books.csv'
df = pd.read_csv(data_path)

# Convert relevant columns to numeric types
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')

# Drop any rows with NaN values in essential columns and standardize 'authors' column
df.dropna(subset=['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count'], inplace=True)
df['authors'] = df['authors'].str.strip()

# Create a ratings matrix for ALS model
ratings_matrix = coo_matrix((df['ratings_count'], (df['bookID'], df['bookID']))).tocsr()

# Fit the ALS recommendation model
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(ratings_matrix)

# Initialize filtered DataFrame for recommendations
filtered_df = pd.DataFrame()

# Get unique authors
unique_authors = sorted(df['authors'].dropna().unique())
# Define rating options
rating_options = [str(i) for i in sorted(df['average_rating'].unique(), reverse=True)]

# Function to filter by selected author
def filter_by_author(author):
    global filtered_df
    filtered_df = df[df['authors'] == author][['title', 'authors', 'average_rating']]
    
    if not filtered_df.empty:
        return f"Books by {author}:\n{filtered_df[['title', 'average_rating']].head(10).to_string(index=False)}"
    else:
        return "No books found for this author."

# Function to filter by selected rating
def filter_by_rating(min_rating):
    global filtered_df
    filtered_df = df[df['average_rating'] >= float(min_rating)][['title', 'authors', 'average_rating']]
    
    if not filtered_df.empty:
        return f"Books with an average rating of {min_rating} or higher:\n{filtered_df[['title', 'average_rating']].head(10).to_string(index=False)}"
    else:
        return f"No books found with an average rating of {min_rating} or higher."

# Get recommendations based on ALS model
def get_recommendations(book_id):
    book_idx = df[df['bookID'] == book_id].index[0]
    recommended_indices, scores = model.recommend(book_idx, ratings_matrix[book_idx], N=5)
    recommended_books = df.iloc[recommended_indices][['title', 'authors', 'average_rating']]
    return f"Recommended Books:\n{recommended_books.to_string(index=False)}"

# Function to display the result in the GUI
def display_result(result):
    result_box.config(state=tk.NORMAL)
    result_box.delete(1.0, tk.END)  # Clear previous content
    result_box.insert(tk.END, result)
    result_box.config(state=tk.DISABLED)

# Function to handle user choices
def handle_choice(choice):
    if choice == "1":
        selected_author = author_dropdown.get()
        if selected_author:
            result = filter_by_author(selected_author)
            display_result(result)
    elif choice == "2":
        selected_rating = rating_dropdown.get()
        if selected_rating:
            result = filter_by_rating(selected_rating)
            display_result(result)
    elif choice == "3":
        book_id = simpledialog.askinteger("Input", "Enter the book ID for recommendations:")
        if book_id is not None:
            result = get_recommendations(book_id)
            display_result(result)
    else:
        messagebox.showerror("Error", "Invalid choice.")

# Initialize main GUI window
root = tk.Tk()
root.title("Book Recommendation System")
root.geometry("700x700")
root.configure(bg='#f0f8ff')  # Light background color

# Add a title label
title_label = tk.Label(root, text="Book Recommendation System", font=('Arial', 18, 'bold'), bg='#f0f8ff')
title_label.pack(pady=10)

# Add option buttons
option_frame = tk.Frame(root, bg='#f0f8ff')
option_frame.pack(pady=10)

button1 = tk.Button(option_frame, text="Get recommendations based on Author", command=lambda: handle_choice("1"), bg='#87ceeb', font=('Arial', 10, 'bold'))
button2 = tk.Button(option_frame, text="Get recommendations based on Average Rating", command=lambda: handle_choice("2"), bg='#87ceeb', font=('Arial', 10, 'bold'))
button3 = tk.Button(option_frame, text="Show model recommendations for a specific book ID", command=lambda: handle_choice("3"), bg='#87ceeb', font=('Arial', 10, 'bold'))

button1.pack(side=tk.LEFT, padx=5)
button2.pack(side=tk.LEFT, padx=5)
button3.pack(side=tk.LEFT, padx=5)

# Author selection dropdown
author_label = tk.Label(root, text="Choose an Author:", font=('Arial', 12, 'bold'), bg='#f0f8ff')
author_label.pack(pady=5)

author_dropdown = ttk.Combobox(root, values=unique_authors, width=60)
author_dropdown.set("Select an author from the list")
author_dropdown.pack(pady=5)

# Rating selection dropdown
rating_label = tk.Label(root, text="Choose a Minimum Average Rating:", font=('Arial', 12, 'bold'), bg='#f0f8ff')
rating_label.pack(pady=5)

rating_dropdown = ttk.Combobox(root, values=rating_options, width=60)
rating_dropdown.set("Select a rating from the list")
rating_dropdown.pack(pady=5)

# Text area for displaying results (increased size for readability)
result_box = scrolledtext.ScrolledText(root, width=80, height=20, bg='#fff', font=('Arial', 12))
result_box.pack(pady=20)
result_box.config(state=tk.DISABLED)

# Run the GUI
root.mainloop()

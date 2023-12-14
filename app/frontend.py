# Import necessary libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Simple Streamlit App")

# Add a slider widget for selecting a number
number = st.slider("Select a number", 1, 10)

# Generate some random data based on the selected number
x = np.linspace(0, 10, 100)
y = np.sin(number * x)

# Create a plot using Matplotlib
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title(f'Sine Wave with {number} cycles')

# Display the plot in the Streamlit app
st.pyplot(fig)

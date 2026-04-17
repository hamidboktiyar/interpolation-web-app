import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Algorithm: Newton's Divided Difference ---
def newton_divided_diff(x, y, xi):
    n = len(x)
    # Initialize the divided difference table
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    # Evaluate the polynomial at xi
    n = len(x) - 1
    p = coef[0][n]
    for k in range(1, n + 1):
        p = coef[0][n - k] + (xi - x[n - k]) * p
    
    return p, coef

# --- Streamlit UI ---
st.set_page_config(page_title="Interpolation Web App", layout="wide")

st.title("📊 Interpolation for Unequal Intervals")
st.subheader("Method: Newton's Divided Difference")

st.write("""
This app implements the Newton's Divided Difference algorithm to find interpolated values 
for datasets where the x-intervals are not uniform.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Input Data")
raw_x = st.sidebar.text_input("Enter X values (comma separated)", "0, 1, 2, 5")
raw_y = st.sidebar.text_input("Enter Y values (comma separated)", "2, 3, 12, 147")
target_x = st.sidebar.number_input("Value to interpolate (X)", value=3.0)

try:
    x_points = np.array([float(i.strip()) for i in raw_x.split(",")])
    y_points = np.array([float(i.strip()) for i in raw_y.split(",")])

    if len(x_points) != len(y_points):
        st.error("Error: Number of X and Y points must match!")
    else:
        # --- Calculation ---
        result, table = newton_divided_diff(x_points, y_points, target_x)

        # --- Display Results ---
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"**Interpolated Value at X = {target_x}:** {result:.4f}")
            
            # Show the Difference Table
            st.write("### Divided Difference Table")
            df_table = pd.DataFrame(table)
            st.dataframe(df_table)

        with col2:
            # --- Visualization ---
            st.write("### Visualization")
            x_range = np.linspace(min(x_points), max(x_points), 100)
            y_range = [newton_divided_diff(x_points, y_points, val)[0] for val in x_range]

            fig, ax = plt.subplots()
            ax.plot(x_range, y_range, label="Interpolated Polynomial", color='blue', alpha=0.6)
            ax.scatter(x_points, y_points, color='red', label="Data Points")
            ax.scatter(target_x, result, color='green', marker='X', s=100, label="Target Result")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            st.pyplot(fig)

except ValueError:
    st.info("Please enter valid numbers separated by commas in the sidebar.")

# --- Footer ---
st.markdown("---")
st.write("Developed for Numerical Mathematics Course Project.")

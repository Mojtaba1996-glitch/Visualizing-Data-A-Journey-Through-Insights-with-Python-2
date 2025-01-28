# Visualizing-Data-A-Journey-Through-Insights-with-Python-2
 This project is dedicated to the art and science of data visualization using Python. In a world driven by data, effective visualization is the bridge between raw information and actionable insights. Through this project, I aim to showcase how complex datasets can be transformed into compelling, interactive, and insightful visual stories.

 ## 📄 Guide
Download the PDF here: [Visualizing Data Guide](docs/docs/docs/Visualizing.pdf)

## 🗂️ Code Examples
### 1. Normalized Stock Prices
![Normalized Prices](figures/normalized_prices.png)
```python
# import pandas pd
import plotly.graph_objects as go

# Function to normalize prices
def normalizedprice(df):
    """
    Normalizes the prices in a DataFrame so they all start at 1.0.
    This makes it easier to compare trends over time.
    """
    normalized_df = df / df.iloc[0]  # Divide each value by the first row's value
    return normalized_df

# Example DataFrame (replace this with your actual data)
# df = pd.read_csv('data/stock_data.csv', index_col='Date', parse_dates=True)

# Normalize the prices
normalized_price = normalizedprice(df)

# Create a Plotly figure
fig = go.Figure()

# Add a line for each column in the normalized DataFrame
for col in normalized_price.columns:
    fig.add_trace(go.Scatter(
        x=normalized_price.index,  # X-axis: Dates
        y=normalized_price[col],   # Y-axis: Normalized prices
        name=col,                  # Name of the line (e.g., stock name)
        mode='lines'               # Draw a line chart
    ))

# Update the layout of the chart
fig.update_layout(
    title='Normalized Stock Prices Over Time',  # Chart title
    xaxis_title='Date',                         # X-axis label
    yaxis_title='Normalized Price',             # Y-axis label
    showlegend=True                             # Show the legend
)

# Save the chart as an image
fig.write_image("figures/normalized_prices.png")

# Display the chart
fig.show()

# Sort and display the normalized prices on a specific date
sorted_prices = normalized_price.loc["2022-09-30"].sort_values(ascending=False)
print("Normalized Prices on 2022-09-30 (Sorted):")
print(sorted_prices)

```

### 2. Portfolio Optimization
![Portfolio Weights](figures/portfolio_optimization.png)
```python
# See code/portfolio_bar.py
```

### 3. Max Drawdown Analysis
![Drawdown Plots](figures/max_drawdown.png)
```python
# See code/drawdown_plots.py
```

### 4. Returns Scatter Plot
![Returns Analysis](figures/returns_analysis.png)
```python
# See code/returns_scatter.py
```

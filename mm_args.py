# order_book_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Functions -------------------------------------------------------------------

def calculate_order_sizes(total_budget, lambda_val, ticks):
    """
    Calculate order sizes across the specified ticks based on exponential weighting.
    """
    distances = np.abs(ticks)  # Distances from mid-price in terms of "tick steps"
    exponential_weights = np.exp(lambda_val * distances)  # Exponential weights
    normalized_weights = exponential_weights / np.sum(exponential_weights)  # Normalize
    order_sizes = total_budget * normalized_weights  # Allocate budget
    return order_sizes

def plot_order_book_flipped(total_budget_bids, 
                            total_budget_asks,
                            lambda_val, 
                            ticks_range,
                            mid_price=0.048, 
                            tick_size=0.00001,
                            spread_width=5):
    """
    Create a flipped ("vertical") order book style plot and return the figure.
    """
    # Separate ticks into bids (negative) and asks (positive), excluding near-mid ticks
    valid_bids_ticks = [t for t in ticks_range if t < 0 and abs(t) >= spread_width]
    valid_asks_ticks = [t for t in ticks_range if t > 0 and abs(t) >= spread_width]

    # If no valid ticks remain on either side, just return an empty figure
    fig, ax = plt.subplots(figsize=(7, 9))
    
    if not valid_bids_ticks and not valid_asks_ticks:
        ax.set_title("No valid bid or ask ticks remain (spread_width too large).")
        return fig

    # Calculate order sizes for bids
    if len(valid_bids_ticks) > 0:
        order_sizes_bids = calculate_order_sizes(total_budget_bids, lambda_val, valid_bids_ticks)
        prices_bids = [mid_price + t * tick_size for t in valid_bids_ticks]
    else:
        order_sizes_bids = np.array([])
        prices_bids = []

    # Calculate order sizes for asks
    if len(valid_asks_ticks) > 0:
        order_sizes_asks = calculate_order_sizes(total_budget_asks, lambda_val, valid_asks_ticks)
        prices_asks = [mid_price + t * tick_size for t in valid_asks_ticks]
    else:
        order_sizes_asks = np.array([])
        prices_asks = []

    # Plot BIDS
    if len(order_sizes_bids) > 0:
        ax.barh(prices_bids, order_sizes_bids, 
                height=tick_size * 0.8, 
                color='blue', edgecolor='black', label='Bids')

        # Add text labels showing both size and price
        for (price, size) in zip(prices_bids, order_sizes_bids):
            ax.text(size, price, f"{size:.2f} @ {price:.5f}", 
                    ha='left', va='center', fontsize=8, color='blue')

    # Plot ASKS
    if len(order_sizes_asks) > 0:
        ax.barh(prices_asks, order_sizes_asks, 
                height=tick_size * 0.8, 
                color='red', edgecolor='black', label='Asks')

        # Add text labels showing both size and price
        for (price, size) in zip(prices_asks, order_sizes_asks):
            ax.text(size, price, f"{size:.2f} @ {price:.5f}",
                    ha='left', va='center', fontsize=8, color='red')

    # Merge all prices for correct y-axis ticks
    all_prices = np.concatenate([prices_bids, prices_asks])
    if len(all_prices) > 0:
        unique_prices = sorted(set(all_prices))
        ax.set_yticks(unique_prices)
        ax.set_yticklabels([f"{p:.5f}" for p in unique_prices])

    # Cosmetics
    ax.set_title("Flipped Order Book", fontsize=14)
    ax.set_xlabel("Order Size", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend()
    
    fig.tight_layout()
    return fig

# --- Streamlit App ---------------------------------------------------------------

def main():
    st.title("Flipped Order Book Visualization")

    st.write("""
    This app lets you visualize how bids and asks are allocated across price levels,
    based on an exponential weighting parameter **lambda** and a chosen **spread width**.
    """)

    # Sidebar parameters
    st.sidebar.header("Order Book Parameters")
    total_budget_bids = st.sidebar.slider('Bids Budget', min_value=0, max_value=5000, value=200, step=50)
    total_budget_asks = st.sidebar.slider('Asks Budget', min_value=0, max_value=5000, value=200, step=50)
    lambda_val = st.sidebar.slider('Lambda', min_value=0.0, max_value=2.0, value=0.25, step=0.05)
    
    mid_price = st.sidebar.number_input("Mid Price", min_value=0.00000, value=0.04800, format="%.5f")
    tick_size = st.sidebar.number_input("Tick Size", min_value=0.0000001, value=0.0001, format="%.7f")
    spread_width = st.sidebar.slider("Spread Width (in ticks)", min_value=1, max_value=50, value=5)

    st.sidebar.header("Tick Range")
    tick_max = st.sidebar.slider("Max Tick Range", min_value=5, max_value=100, value=15)
    ticks_range = np.arange(-tick_max, tick_max + 1)

    # Generate and display plot
    fig = plot_order_book_flipped(
        total_budget_bids,
        total_budget_asks,
        lambda_val,
        ticks_range,
        mid_price=mid_price,
        tick_size=tick_size,
        spread_width=spread_width
    )
    st.pyplot(fig)

if __name__ == "__main__":
    main()

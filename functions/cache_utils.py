import os
import json
import pickle
from datetime import datetime, timedelta

# In-memory cache
plot_cache = {}

# File to store cache data
CACHE_FILE = 'data/plot_cache.pkl'

def update_plot_cache(plot_key, fig, expiration_hours=24):
    """
    Update the plot cache with a new figure.
    
    :param plot_key: A unique identifier for the plot (e.g., 'daily_pm25')
    :param fig: The plotly figure object
    :param expiration_hours: Number of hours before the cache expires
    """
    expiration_time = datetime.now() + timedelta(hours=expiration_hours)
    plot_cache[plot_key] = {
        'fig': fig,
        'expiration': expiration_time
    }
    save_cache_to_file()

def get_cached_plot(plot_key):
    """
    Retrieve a plot from the cache if it exists and hasn't expired.
    
    :param plot_key: The unique identifier for the plot
    :return: The cached figure if available, else None
    """
    if plot_key in plot_cache:
        cache_entry = plot_cache[plot_key]
        if datetime.now() < cache_entry['expiration']:
            return cache_entry['fig']
        else:
            del plot_cache[plot_key]
    return None

def save_cache_to_file():
    """Save the current cache to a file."""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(plot_cache, f)

def load_cache_from_file():
    """Load the cache from a file if it exists."""
    global plot_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            plot_cache = pickle.load(f)

# Load cache when the module is imported
load_cache_from_file()

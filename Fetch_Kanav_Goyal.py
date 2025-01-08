#!/usr/bin/env python
# coding: utf-8

# In[2584]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# LOAD AND PREPROCESS DATA
###############################################################################

df = pd.read_csv("https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv",parse_dates=['# Date'])


# In[2586]:


df.head()


# In[ ]:





# In[2589]:


# Set the "# Date" column as the index, converting it to a PeriodIndex for better handling of time periods
df = df.set_index("# Date").to_period()

# Ensure the data has a daily frequency, filling in any missing dates as needed
df = df.asfreq('D')


# In[ ]:





# In[2592]:


# Calculate a 365-day moving average for the receipt counts
moving_average = df.rolling(
    window=365,       # Define a rolling window of 365 days
    center=True,      # Center the average within the window for alignment
    min_periods=180,  # Require at least 180 valid data points to compute the average
).mean()              # Compute the mean for each rolling window

# Plot the original daily receipt counts
ax = df.plot(style=".", color="0.5")  # Use dots with a light gray color for better visibility of raw data

# Overlay the moving average on the same plot
moving_average.plot(
    ax=ax, linewidth=3,  # Thicker line for the moving average to stand out
    title="Daily Receipt Counts with 365-Day Moving Average",  # Add a meaningful title
    legend=False,  # Disable the legend for simplicity
)

# Save the plot as a high-resolution PNG file
plt.savefig('/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/Daily Receipt Counts with 365-Day Moving Average.png', 
            dpi=300,  # Set a high DPI for better image quality
            bbox_inches='tight')  # Ensure no labels or parts of the graph are cut off

# Display the plot
plt.show()


# In[2593]:


# Calculate a 180-day moving average for the receipt counts
moving_average = df.rolling(
    window=180,       # Define a rolling window of 180 days
    center=True,      # Center the average within the window for alignment
    min_periods=90,   # Require at least 90 valid data points to compute the average
).mean()              # Compute the mean for each rolling window

# Plot the original daily receipt counts
ax = df.plot(
    style=".",        # Use dots to represent raw data points
    color="0.5"       # Light gray color for raw data to make it visually subtle
)

# Overlay the moving average on the same plot
moving_average.plot(
    ax=ax,            # Plot on the same axis as the raw data
    linewidth=3,      # Thicker line for the moving average to stand out
    title="Daily Receipt Counts with 180-Day Moving Average",  # Add a descriptive title
    legend=False      # Disable the legend for simplicity
)

# Save the graph as a high-resolution PNG file
plt.savefig(
    '/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/Daily Receipt Counts with 180-Day Moving Average.png',  # File path for the image
    dpi=300,          # High DPI for quality output
    bbox_inches='tight'  # Ensure no labels or plot elements are cut off
)

# Display the plot
plt.show()            # Render the plot in the output window


# In[2594]:


from statsmodels.tsa.deterministic import DeterministicProcess

# Create a deterministic process for time-based features
dp = DeterministicProcess(
    index=df.index,  # Use the index from the DataFrame as the time points
    constant=False,  # Exclude a constant (intercept) term; no dummy for bias
    order=2,         # Include terms for a linear and quadratic trend (time-based)
    drop=True,       # Automatically drop terms to avoid collinearity if necessary
)

# Generate features for the given dates in the index
X = dp.in_sample()  # Creates time-based features for the provided index

# Join the generated features (e.g., trend, trend squared) to the original DataFrame
df = df.join(X)

# Display the updated DataFrame with the new time-based features
df


# In[2595]:


from pathlib import Path  # For working with file paths
from warnings import simplefilter  # To filter out warnings

import matplotlib.pyplot as plt  # For creating plots
import pandas as pd  # For handling and analyzing data
import seaborn as sns  # For creating aesthetically pleasing plots
from sklearn.linear_model import LinearRegression  # For simple linear regression modeling
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess  # For time-series modeling

# Define a function to create seasonal plots
def seasonal_plot(X, y, period, freq, ax=None):
    """
    Creates a seasonal plot for time-series data.

    Parameters:
    - X: DataFrame containing features for plotting
    - y: Target variable (e.g., receipt counts) to be plotted
    - period: The seasonal period (e.g., 'week', 'month')
    - freq: The frequency within the period (e.g., 'dayofweek', 'dayofmonth')
    - ax: (Optional) Axes object to plot on; creates a new one if None
    """
    # Create a new subplot if no Axes object is provided
    if ax is None:
        _, ax = plt.subplots()

    # Define a color palette for the plot based on the number of unique periods
    palette = sns.color_palette("husl", n_colors=X[period].nunique())

    # Create a line plot using seaborn
    ax = sns.lineplot(
        x=freq,          # The frequency (x-axis)
        y=y,             # The target variable (y-axis)
        hue=period,      # Use the period to differentiate lines (e.g., week, month)
        data=X,          # The DataFrame containing the data
        ci=False,        # Disable confidence intervals for cleaner lines
        ax=ax,           # Plot on the specified or new Axes object
        palette=palette, # Use the defined color palette
        legend=False,    # Disable the legend for simplicity
    )

    # Add a title indicating the type of seasonal plot
    ax.set_title(f"Seasonal Plot ({period}/{freq})")

    # Annotate the end of each line with the corresponding period name
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]  # Get the last y-value of the line
        ax.annotate(
            name,                  # Label with the period name
            xy=(1, y_),            # Position the label at the end of the line
            xytext=(6, 0),         # Offset the label slightly to the right
            color=line.get_color(),# Match the label color to the line color
            xycoords=ax.get_yaxis_transform(),  # Use axis-relative coordinates
            textcoords="offset points",        # Position the label in offset points
            size=14,              # Set the font size for the label
            va="center",          # Vertically align the label at the center
        )

    # Return the Axes object for further customization if needed
    return ax


# In[2596]:


# Create a copy of the DataFrame to add new features for seasonal analysis
X = df.copy()

# Add features for weekly seasonality
X["day"] = X.index.dayofweek  # Day of the week (0=Monday, 6=Sunday) for the x-axis (freq)
X["week"] = X.index.week      # ISO week number (1-52) to represent the seasonal period (period)

# Add features for yearly seasonality
X["dayofyear"] = X.index.dayofyear  # Day of the year (1-365/366) for x-axis (freq)
X["year"] = X.index.year            # Year component to represent the seasonal period (period)

# Create subplots for weekly and yearly seasonal plots
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))  # Two rows, one column of plots

# Plot weekly seasonality
seasonal_plot(X, y="Receipt_Count", period="week", freq="day", ax=ax0)

# Save the weekly seasonal plot as a high-resolution PNG
plt.savefig(
    '/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/seasonal_plot1.png', 
    dpi=300, 
    bbox_inches='tight'  # Ensure no part of the plot is cut off
)

# Plot yearly seasonality
seasonal_plot(X, y="Receipt_Count", period="year", freq="dayofyear", ax=ax1)

# Save the yearly seasonal plot as a high-resolution PNG
plt.savefig(
    '/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/seasonal_plot2.png', 
    dpi=300, 
    bbox_inches='tight'
)

# Add features for monthly seasonality
X["dayofmonth"] = X.index.day   # Day of the month (1-31) for x-axis (freq)
X["month"] = X.index.month      # Month of the year (1-12) to represent the seasonal period (period)

# Create a plot for monthly seasonality
fig, ax = plt.subplots(1, 1, figsize=(11, 6))  # Single plot for monthly seasonality

# Plot monthly seasonality
seasonal_plot(X, y="Receipt_Count", period="month", freq="dayofmonth", ax=ax)

# Save the monthly seasonal plot as a high-resolution PNG
plt.savefig(
    '/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/seasonal_plot3.png', 
    dpi=300, 
    bbox_inches='tight'
)

# Display all plots in the output window
plt.show()


# In[2597]:


df


# In[2598]:


X


# In[ ]:





# In[2600]:


# Step 1: Convert the PeriodIndex to a DatetimeIndex
X.index = X.index.to_timestamp()  # Converts the index from PeriodIndex to DatetimeIndex for compatibility

# Add a new column indicating the name of the day of the week
X['day_of_week'] = X.index.day_name()  # Converts day numbers (e.g., 0, 1) to names (e.g., 'Monday', 'Tuesday')

# Step 2: One-hot encode the 'day_of_week' column
one_hot = pd.get_dummies(X['day_of_week'], prefix='day')  # Create binary columns for each day of the week
one_hot = one_hot.astype(int)  # Ensure all values are integers (0 or 1)

# Step 3: Drop one column to avoid multicollinearity in regression models (optional)
one_hot = one_hot.drop(columns=['day_Monday'])  # Drop 'Monday' as the baseline (reference category)

# Step 4: Add the one-hot encoded columns to the original dataframe
X = pd.concat([X, one_hot], axis=1)  # Concatenate the one-hot encoded columns with the original dataframe

# Display the first few rows of the updated dataframe
X.head()  # View the dataframe to confirm the new columns have been added


# In[2601]:


from pathlib import Path  # For working with file paths
from warnings import simplefilter  # To manage warnings

import matplotlib.pyplot as plt  # For creating visualizations
import numpy as np  # For numerical computations
import pandas as pd  # For working with datasets
import seaborn as sns  # For creating visually appealing plots
from scipy.signal import periodogram  # For frequency domain analysis
from sklearn.linear_model import LinearRegression  # For regression modeling
from sklearn.model_selection import train_test_split  # For splitting datasets
from statsmodels.graphics.tsaplots import plot_pacf  # For plotting partial autocorrelation functions

# Define a function to create a lag plot
def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    """
    Create a lag plot to visualize the relationship between a variable and its lagged version.

    Parameters:
    - x: Series or DataFrame column to be lagged
    - y: Optional; another Series to plot against lagged x (default is None, using x itself)
    - lag: Number of lag steps to apply (default is 1)
    - standardize: Whether to standardize the data for easier comparison
    - ax: Optional; Axes object to plot on
    - **kwargs: Additional keyword arguments for `sns.regplot`

    Returns:
    - ax: The Axes object containing the plot
    """
    from matplotlib.offsetbox import AnchoredText  # For adding correlation annotations
    
    # Shift the x variable by the specified lag
    x_ = x.shift(lag)
    
    # Standardize the lagged and optional y variable if required
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x

    # Calculate the correlation coefficient between lagged x and y
    corr = y_.corr(x_)

    # Create a new Axes object if none is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Define default styles for scatter points and regression line
    scatter_kws = dict(
        alpha=0.75,  # Transparency level for scatter points
        s=3,         # Size of scatter points
    )
    line_kws = dict(color='C3')  # Color for the regression line

    # Create the scatter plot with a lowess-smoothed regression line
    ax = sns.regplot(
        x=x_,
        y=y_,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        lowess=True,  # Locally Weighted Regression
        ax=ax,
        **kwargs
    )

    # Add the correlation coefficient as an annotation on the plot
    at = AnchoredText(
        f"{corr:.2f}",  # Display the correlation coefficient rounded to 2 decimal places
        prop=dict(size="large"),
        frameon=True,  # Add a background box
        loc="upper left",  # Position the annotation in the upper-left corner
    )
    at.patch.set_boxstyle("square, pad=0.0")  # Set the annotation box style
    ax.add_artist(at)

    # Set titles and labels
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


# Define a function to create multiple lag plots
def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    """
    Create a grid of lag plots for multiple lags.

    Parameters:
    - x: Series or DataFrame column to be lagged
    - y: Optional; another Series to plot against lagged x
    - lags: Total number of lags to plot (default is 6)
    - nrows: Number of rows in the grid (default is 1)
    - lagplot_kwargs: Additional arguments to pass to `lagplot`
    - **kwargs: Additional keyword arguments for `plt.subplots`

    Returns:
    - fig: The Figure object containing the lag plots
    """
    import math  # For mathematical operations

    # Set default values for the subplot grid
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))  # Calculate the number of columns needed
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))  # Adjust the figure size

    # Create the subplot grid
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)

    # Loop through each subplot and plot the corresponding lag
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            # Plot the lag for the current axis
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            # Turn off unused axes
            ax.axis('off')

    # Set labels for the last row and first column
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)

    # Adjust layout to prevent overlapping elements
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


# In[2602]:


# Create a grid of lag plots to analyze temporal relationships in the data
_ = plot_lags(
    X.Receipt_Count,  # The target variable (Receipt Count) to analyze
    lags=12,          # Number of lags to include in the plot
    nrows=2           # Arrange the plots in 2 rows
)

# Save the lag plot grid as a high-resolution PNG file
plt.savefig(
    '/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/lagplot.png',  # File path for the saved image
    dpi=300,          # High DPI ensures the image is clear and professional
    bbox_inches='tight'  # Prevents cutting off any part of the plot
)

# Create a Partial Autocorrelation Function (PACF) plot
_ = plot_pacf(
    X.Receipt_Count,  # The target variable (Receipt Count) for PACF analysis
    lags=12           # Number of lags to include in the PACF
)

# Save the PACF plot as a high-resolution PNG file
plt.savefig(
    '/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/PA.png',  # File path for the saved PACF plot
    dpi=300,          # High DPI ensures clear visualization
    bbox_inches='tight'  # Ensures no labels or titles are cut off
)


# In[2604]:


def make_lags(ts, lags):
    return pd.concat(
        {f'y_lag_{i}': ts.shift(i) for i in range(1, lags + 1)}, axis=1
    )

# Generate lag features
lag_features = make_lags(X.Receipt_Count, lags=5)

# Fill NaN values with 0.0 for the lagged features
lag_features = lag_features.fillna(0.0)

# Include lag features in X
X = pd.concat([X, lag_features], axis=1)


# In[2605]:


X


# In[2606]:


# X.columns
# X=X[['trend', 'trend_squared','week','dayofmonth', 'month', 'day_Friday',
#        'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday',
#        'day_Wednesday', 'y_lag_1', 'y_lag_2', 'y_lag_3', 'y_lag_4','y_lag_5']]

X.columns
X=X[['trend','week','dayofmonth', 'month', 'day_Friday',
       'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday',
       'day_Wednesday']]


# In[2607]:


X


# In[ ]:





# In[ ]:





# In[ ]:





# In[2611]:


def manual_train_test_split(X, y, test_size, shuffle=False, random_state=None):
    """
    Manually split arrays into random train and test subsets.
    Preserves pandas DataFrame/Series types and indices if provided.
    
    Parameters:
    -----------
    X : array-like, pandas DataFrame, or pandas Series
        Features dataset
    y : array-like or pandas Series
        Target dataset
    test_size : int or float
        If float, represents proportion of dataset to include in test split (0.0 to 1.0)
        If int, represents absolute number of test samples
    shuffle : boolean, default=False
        Whether to shuffle the data before splitting
    random_state : int, default=None
        Controls the shuffling applied to the data
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : same type as inputs
        The split datasets
    """
    # Store input types
    X_is_pandas = isinstance(X, (pd.DataFrame, pd.Series))
    y_is_pandas = isinstance(y, pd.Series)
    
    # Convert inputs to numpy arrays for processing
    X_values = X.values if X_is_pandas else np.array(X)
    y_values = y.values if y_is_pandas else np.array(y)
    
    # Calculate number of test samples
    n_samples = len(X_values)
    if isinstance(test_size, float):
        n_test = int(test_size * n_samples)
    else:
        n_test = test_size
    
    # Calculate number of training samples
    n_train = n_samples - n_test
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    # Split the indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split the data
    X_train_values = X_values[train_indices]
    X_test_values = X_values[test_indices]
    y_train_values = y_values[train_indices]
    y_test_values = y_values[test_indices]
    
    # Convert back to original types if input was pandas
    if X_is_pandas:
        if isinstance(X, pd.DataFrame):
            X_train = pd.DataFrame(X_train_values, index=X.index[train_indices], columns=X.columns)
            X_test = pd.DataFrame(X_test_values, index=X.index[test_indices], columns=X.columns)
        else:  # Series
            X_train = pd.Series(X_train_values, index=X.index[train_indices], name=X.name)
            X_test = pd.Series(X_test_values, index=X.index[test_indices], name=X.name)
    else:
        X_train = X_train_values
        X_test = X_test_values
        
    if y_is_pandas:
        y_train = pd.Series(y_train_values, index=y.index[train_indices], name=y.name)
        y_test = pd.Series(y_test_values, index=y.index[test_indices], name=y.name)
    else:
        y_train = y_train_values
        y_test = y_test_values
    
    return X_train, X_test, y_train, y_test


# In[2612]:


class ManualStandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    The standard score of a sample x is calculated as: z = (x - u) / s
    where u is the mean of the training samples, and s is the standard deviation.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features = None
        
    def fit(self, X):
        """
        Compute the mean and standard deviation of X for later scaling.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
        """
        # Convert to numpy array if needed
        X = np.array(X)
        
        # Handle both 1D and 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features = X.shape[1]
        
        # Calculate mean and standard deviation for each feature
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1)  # ddof=1 for sample standard deviation
        
        # Handle constant features
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to transform
        
        Returns:
        --------
        X_scaled : array-like of shape (n_samples, n_features)
            The transformed data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        # Convert to numpy array if needed
        X = np.array(X)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(-1, 1)
            
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            
        # Perform standardization
        X_scaled = (X - self.mean_) / self.scale_
        
        # Convert back to pandas if input was pandas
        if isinstance(X, pd.DataFrame):
            X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        elif isinstance(X, pd.Series):
            X_scaled = pd.Series(X_scaled.ravel(), index=X.index, name=X.name)
        elif is_1d:
            X_scaled = X_scaled.ravel()
            
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to inverse transform
            
        Returns:
        --------
        X_orig : array-like of shape (n_samples, n_features)
            The original data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        # Convert to numpy array if needed
        X = np.array(X)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(-1, 1)
            
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            
        # Perform inverse transform
        X_orig = X * self.scale_ + self.mean_
        
        # Convert back to pandas if input was pandas
        if isinstance(X, pd.DataFrame):
            X_orig = pd.DataFrame(X_orig, index=X.index, columns=X.columns)
        elif isinstance(X, pd.Series):
            X_orig = pd.Series(X_orig.ravel(), index=X.index, name=X.name)
        elif is_1d:
            X_orig = X_orig.ravel()
            
        return X_orig


# In[2613]:


import numpy as np
import pandas as pd


class BetterLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.X_scaler = ManualStandardScaler()
        self.y_scaler = ManualStandardScaler()
        
    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale the data
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Add bias term to X
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        
        # Calculate weights using Normal Equation
        # w = (X^T X)^(-1) X^T y
        try:
            # Using pseudo-inverse for better numerical stability
            weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_scaled)
            
            self.bias = weights[0]
            self.weights = weights[1:]
            
            # Calculate and print R-squared
            y_pred_scaled = self.predict_scaled(X_scaled)
            
        except np.linalg.LinAlgError:
            print("Error: Matrix is singular or nearly singular")
            return None
            
    def predict_scaled(self, X_scaled):
        return np.dot(X_scaled, self.weights) + self.bias
        
    def predict(self, X):
        X_scaled = self.X_scaler.transform(np.array(X))
        y_scaled_pred = self.predict_scaled(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()


# Split the data
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=60, shuffle=False)

# Train the model
model = BetterLinearRegression()
model.fit(X_train, y_train)

# Generate predictions
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)


# In[ ]:





# In[ ]:





# In[2619]:


# Plot the data
ax = y_train.plot(label="Actual (Observed - Training Data)", color="black", style="--")
ax = y_test.plot(label="Actual (Observed - Test Data)", color="black", linestyle=":")
ax = y_pred.plot(ax=ax, label="Predicted (Training Data)", color="blue", linestyle="--")
_ = y_fore.plot(ax=ax, label="Forecast (Test Data)", color="red", linestyle="--")

# Add titles and axis labels
ax.set_title("Actual vs Predicted Receipt Counts with Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Receipt Counts")

# Add legend
ax.legend()

# Show the plot
plt.savefig('/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/Actual vs Predicted Receipt Counts with Forecast.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
plt.show()


# In[ ]:





# In[ ]:





# In[2622]:


# ax = y_test.plot(**plot_params)
# _ = y_fore.plot(ax=ax, color='C3')

# Plot the actual test data
ax = y_test.plot(
    label="Actual (Observed - Test Data)", 
    linestyle=":", 
    alpha=0.7,  # Slight transparency for better visibility
    **plot_params  # Use color and other settings from plot_params
)

# Plot the forecast
_ = y_fore.plot(
    ax=ax, 
    label="Forecast (Test Data)", 
    color='C3',  # Red color for forecast line
    linestyle="--", 
    linewidth=2  # Thicker line for better visibility
)

# Add titles and axis labels
ax.set_title("Actual vs Forecast Receipt Counts", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Receipt Counts", fontsize=12)

# Add legend
ax.legend(
    loc="upper left", 
    fontsize=10, 
    title="Legend", 
    title_fontsize=12, 
    frameon=True, 
    framealpha=0.9  # Add a semi-transparent background to the legend
)

# Add a grid for better readability
plt.grid(True, linestyle="--", alpha=0.6)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.savefig('/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/Actual vs Forecast Receipt Counts.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
plt.show()


# In[2623]:


import joblib

# Save the trained model
joblib.dump(model, "/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/model_updated.pkl")  # Replace 'model' with your trained model variable


# In[2624]:


def create_features(date):
    """Create features for a single date"""
    # Convert input date to Timestamp if it's not already
    date = pd.Timestamp(date)
    
    df = pd.DataFrame(index=[date])
    
    # Add trend features
    dp = pd.DataFrame(index=[date])
    days_since_start = (date - pd.Timestamp('2021-01-01')).days
    dp['trend'] = days_since_start
    # dp['trend_squared'] = days_since_start ** 2
    
    # Add time-based features
    df['week'] = date.isocalendar()[1]
    df['dayofmonth'] = date.day
    df['month'] = date.month
    
    # Add day of week dummies
    day_name = date.day_name()
    days = ['Friday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    for day in days:
        df[f'day_{day}'] = 1 if day_name == day else 0
        
    # Combine all features
    df = pd.concat([df, dp], axis=1)
    
    # Ensure correct column order
    columns = ['trend', 'trend_squared', 'week', 'dayofmonth', 'month', 
               'day_Friday', 'day_Saturday', 'day_Sunday', 'day_Thursday', 
               'day_Tuesday', 'day_Wednesday']
    
    return df[columns]


# In[ ]:





# In[ ]:





# In[2627]:


# import pandas as pd

# # 1. Create the 2022 date range
# dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# # 2. Call create_features(date) for each date, then concatenate
# df_2022 = pd.concat([create_features(day) for day in dates_2022])

# # 3. Now df_2022 has one row per day in 2022, including the correct trend
# print(df_2022.head())
# print(df_2022.tail())


# In[2628]:


# # Train the model on all 2021 data
# model = BetterLinearRegression()
# model.fit(X, df['Receipt_Count'])

# # Generate predictions for 2022
# future_predictions = pd.Series(
#     model.predict(future_X), 
#     index=future_X.index
# )

# # If you want monthly predictions
# monthly_predictions = future_predictions.resample('M').sum()


# In[2629]:


# df_2022.columns
# df_2022=df_2022[['trend', 'week', 'dayofmonth', 'month', 'day_Friday',
#        'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday',
#        'day_Wednesday']]


# In[2630]:


# future_predictions = pd.Series(
#     model.predict(df_2022), 
#     index=df_2022.index)


# In[ ]:





# In[ ]:





# In[2633]:


# df=create_features('2022-11-11')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





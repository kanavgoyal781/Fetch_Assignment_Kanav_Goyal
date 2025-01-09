#!/usr/bin/env python
# coding: utf-8

# In[27]:





# In[ ]:





# In[30]:





# In[36]:





# In[45]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the StandardScaler and LinearRegression class definitions
class ManualStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features = None
        
    def fit(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self
    
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        X = np.array(X)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        X_scaled = (X - self.mean_) / self.scale_
        if isinstance(X, pd.DataFrame):
            X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        elif isinstance(X, pd.Series):
            X_scaled = pd.Series(X_scaled.ravel(), index=X.index, name=X.name)
        elif is_1d:
            X_scaled = X_scaled.ravel()
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        X = np.array(X)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        X_orig = X * self.scale_ + self.mean_
        if isinstance(X, pd.DataFrame):
            X_orig = pd.DataFrame(X_orig, index=X.index, columns=X.columns)
        elif isinstance(X, pd.Series):
            X_orig = pd.Series(X_orig.ravel(), index=X.index, name=X.name)
        elif is_1d:
            X_orig = X_orig.ravel()
        return X_orig

class BetterLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.X_scaler = ManualStandardScaler()
        self.y_scaler = ManualStandardScaler()
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        try:
            weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_scaled)
            self.bias = weights[0]
            self.weights = weights[1:]
        except np.linalg.LinAlgError:
            print("Error: Matrix is singular or nearly singular")
            return None
            
    def predict_scaled(self, X_scaled):
        return np.dot(X_scaled, self.weights) + self.bias
        
    def predict(self, X):
        X_scaled = self.X_scaler.transform(np.array(X))
        y_scaled_pred = self.predict_scaled(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()

def create_features(date):
    """Create features for a single date"""
    date = pd.Timestamp(date)
    
    df = pd.DataFrame(index=[date])
    
    days_since_start = (date - pd.Timestamp('2021-01-01')).days
    df['trend'] = days_since_start
    # df['trend_squared'] = days_since_start ** 2
    
    df['week'] = date.isocalendar()[1]
    df['dayofmonth'] = date.day
    df['month'] = date.month
    
    day_name = date.day_name()
    days = ['Friday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    for day in days:
        df[f'day_{day}'] = 1 if day_name == day else 0
    
    columns = ['trend', 'week', 'dayofmonth', 'month', 
               'day_Friday', 'day_Saturday', 'day_Sunday', 'day_Thursday', 
               'day_Tuesday', 'day_Wednesday']
    
    return df[columns]

def load_model():
    try:
        model = joblib.load("model_updated.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' exists in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Receipt Count Predictor")
    st.write("This app predicts receipt counts based on temporal features.")
    
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.stop()
    else:
        st.success("Model loaded successfully! Hurray")



    
    st.write("### Make Predictions")
    date_input = st.date_input(
        "Select a date:",
        min_value=pd.Timestamp('2022-01-01'),
        max_value=pd.Timestamp('2022-12-31'),
        value=pd.Timestamp('2022-01-01')
    )
    
    if st.button("Predict"):
        try:
            features = create_features(date_input)
            prediction = model.predict(features)[0]
            
            st.write("### Prediction Results")
            st.write(f"Date: {date_input.strftime('%Y-%m-%d')}")
            st.write(f"Predicted Receipt Count: {prediction:,.2f}")
            
            if st.checkbox("Show feature values"):
                st.write("### Feature Values Used")
                st.dataframe(features)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please ensure the input date is valid and try again.")
    st.write("Oh! Don't forget to see the Visualisations :)")
    st.title("Visualisations")
    st.write("Time for some trend analysis")
    st.image('daily_receipt_counts_2021.png', use_container_width=True)
    st.image('Daily Receipt Counts with 365-Day Moving Average.png', use_container_width=True)
    st.image('Daily Receipt Counts with 180-Day Moving Average.png',  use_container_width=True)
    st.write("Time for some Seasonality check")
    st.image('seasonal_plot2.png',  use_container_width=True)
    st.image('seasonal_plot3.png',  use_container_width=True)
    st.write("Not being used, but I just made it, so I will just add it too.")
    st.image('lagplot.png', use_container_width=True)
    st.image('PA.png',  use_container_width=True)
    st.image('Actual vs Predicted Receipt Counts with Forecast.png', use_container_width=True)
    st.image('Actual vs Forecast Receipt Counts.png',  use_container_width=True)
    st.write("Thanks for having a look at my app. You have reached the end. Have a good one!")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





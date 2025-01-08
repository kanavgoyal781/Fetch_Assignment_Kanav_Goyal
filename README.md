# Fetch_Assignment_Kanav_Goyal
Description
This application predicts receipt counts based on temporal features using a pre-trained model. It includes visualizations and user input for predictions.


Prerequisites
1. **Python Version**: Ensure you have Python 3.7 or higher installed.
2. **Dependencies**: Install the required libraries by running:
   pip install streamlit pandas numpy seaborn joblib matplotlib

Folder Structure
Your folder should contain the following files:

Fetch/
├── model_updated.pkl                              # Pre-trained model file
├── daily_receipt_counts_2021.png                 # Visualization 1
├── Daily Receipt Counts with 365-Day Moving Average.png   # Visualization 2
├── Daily Receipt Counts with 180-Day Moving Average.png   # Visualization 3
├── seasonal_plot1.png                            # Weekly seasonal plot
├── seasonal_plot2.png                            # Yearly seasonal plot
├── seasonal_plot3.png                            # Monthly seasonal plot
├── lagplot.png                                   # Lag plot
├── PA.png                                        # Partial Autocorrelation plot
├── Actual vs Predicted Receipt Counts with Forecast.png  # Forecast visualization
├── Actual vs Forecast Receipt Counts.png         # Additional forecast visualization
└── Stream_Lit_App_Kanav.py                       # The Streamlit app file

How to Set Up and Run
1. **Download the Repository:**
- Clone this repository or download the files manually into a folder named `Fetch`.
2. **Update File Paths in the Code:**
- Replace the local file paths in `Stream_Lit_App_Kanav.py` to point to the correct file locations on your system.
- **Example:**
  Replace:
  st.image('/Users/kanav/Desktop/MS UChicago/Internship Info/Fetch/daily_receipt_counts_2021.png', use_column_width=True)
  
  With:
  
  st.image('daily_receipt_counts_2021.png', use_column_width=True)

3. **Run the Application:**
- Navigate to the folder containing the `Stream_Lit_App_Kanav.py` file.
- Run the app using Streamlit:
  streamlit run Stream_Lit_App_Kanav.py

Notes on File Paths
- All image files and the model (`model_updated.pkl`) must be placed in the **same directory** as the Streamlit script (`Stream_Lit_App_Kanav.py`).

Contact for Support
If there is any issue contact - kanavgoyal@uchicago.edu or +13122872109
The app runs well. 


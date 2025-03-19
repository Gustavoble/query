import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set the title of the app
st.title("Weather Data Analysis for Crop Yield Prediction")

# File uploader component
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.subheader("Dataset Preview:")
    st.dataframe(data.head())

    # Calculate the maximum and minimum values for selected attributes
    max_values = data[['Humidity_pct', 'Temperature_C', 'Precipitation_mm', 'Wind_Speed_kmh']].max()
    min_values = data[['Humidity_pct', 'Temperature_C', 'Precipitation_mm', 'Wind_Speed_kmh']].min()

    # Display maximum and minimum values
    st.subheader("Maximum Values:")
    st.write(max_values)

    st.subheader("Minimum Values:")
    st.write(min_values)

    # Function to categorize crop yield based on weather attributes
    def categorize_yield(row):
        score = 0  # Initialize score

        # Temperature check
        if 15 <= row['Temperature_C'] <= 25:
            score += 1  # Good
        elif 10 <= row['Temperature_C'] < 15 or 25 < row['Temperature_C'] <= 30:
            score += 0  # Moderate
        else:
            score -= 1  # Bad

        # Humidity check
        if 60 <= row['Humidity_pct'] <= 80:
            score += 1  # Good
        elif 40 <= row['Humidity_pct'] < 60 or 80 < row['Humidity_pct'] <= 90:
            score += 0  # Moderate
        else:
            score -= 1  # Bad

        # Precipitation check
        if 10 <= row['Precipitation_mm'] <= 12:
            score += 1  # Good
        elif 5 <= row['Precipitation_mm'] < 10 or 12 < row['Precipitation_mm'] <= 14:
            score += 0  # Moderate
        else:
            score -= 1  # Bad

        # Wind Speed check
        if row['Wind_Speed_kmh'] <= 10:
            score += 1  # Good
        elif 10 < row['Wind_Speed_kmh'] <= 20:
            score += 0  # Moderate
        else:
            score -= 1  # Bad

        # Final results based on score
        if score >= 3:
            return "Good"
        elif score == 2:
            return "Moderate"
        else:
            return "Bad"

    # Apply the function to create the new 'Crop_Yield' column
    data['Crop_Yield'] = data.apply(categorize_yield, axis=1)

    # Display results
    st.subheader("Updated Dataset with Crop Yield:")
    st.dataframe(data.head())

    # Count the occurrences of each crop yield category
    yield_counts = data['Crop_Yield'].value_counts()

    # Display the counts for each category
    st.subheader("Count of Crop Yield Categories:")
    st.write(yield_counts)

    # Group the data by 'Location' and count occurrences of each crop yield category
    location_yield_counts = data.groupby('Location')['Crop_Yield'].value_counts().unstack(fill_value=0)

    # Display the counts for each location
    st.subheader("Count of Crop Yield Categories by Location:")
    st.dataframe(location_yield_counts)

    # Convert the 'Date' column to datetime format
    data['Date_Time'] = pd.to_datetime(data['Date_Time'], format='%m/%d/%Y')

    # Extract the year and month from the 'Date' column
    data['YearMonth'] = data['Date_Time'].dt.to_period('M')

    # Group the data by 'Location' and 'YearMonth' and count occurrences of each crop yield category
    monthly_location_yield_counts = (
        data.groupby(['Location', 'YearMonth'])['Crop_Yield']
        .value_counts()
        .unstack(fill_value=0)
    )

    # Display the counts for each location and month
    st.subheader("Count of Crop Yield Categories by Location and Month:")
    st.dataframe(monthly_location_yield_counts)

    # Convert categorical crop yield to numerical values
    def yield_to_numeric(yield_category):
        if yield_category == "Good":
            return 1
        elif yield_category == "Moderate":
            return 0
        elif yield_category == "Bad":
            return -1
        return None

    data['Crop_Yield_Numeric'] = data['Crop_Yield'].apply(yield_to_numeric)

    # Drop any rows with missing values in the relevant columns
    relevant_columns = ['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh', 'Crop_Yield_Numeric']
    data_subset = data[relevant_columns].dropna()

    # Calculate the correlation matrix
    correlation_matrix = data_subset.corr()

    # Display the correlation matrix
    st.subheader("Correlation Matrix:")
    st.write(correlation_matrix)

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix between Weather Attributes and Crop Yield')
    st.pyplot(plt)

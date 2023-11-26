import numpy as np
import pickle
import streamlit as st
import warnings
import folium
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)



# Add HTML and CSS to set the background image



# Rest of your Streamlit app code goes here



st.title("WEATHER PREDICTION DASHBOARD")
st.subheader("What is Weather Prediction?")
st.write("Weather prediction, also known as weather forecasting, is the process of using scientific methods "
         "to estimate the state of the atmosphere at a specific location and time in the future. "
         "It involves analyzing various meteorological data, such as temperature, humidity, "
         "wind speed, and atmospheric pressure, to make predictions about weather conditions "
         "like rain, snow, storms, and temperature changes.")
print("\n")
st.subheader("Role of Machine Learning in weather prediction")
st.write("Machine learning has transformed weather prediction by efficiently analyzing vast datasets, "
         "identifying patterns, and uncovering subtle correlations in historical and real-time weather data."
         " This technology enhances accuracy, enabling early warnings for extreme events like hurricanes. "
         "Ensemble forecasting, employing multiple simulations with varied initial conditions, "
         "improves prediction reliability, especially for uncertain weather events. "
         "Machine learning also enables nowcasting, providing immediate short-term forecasts crucial for"
         " decision-making. Additionally, customized forecasts tailored to specific regions "
         "enhance accuracy, benefiting sectors like agriculture and disaster preparedness. "
         "These advancements significantly impact various industries, improving weather forecasts"
         " and aiding in crucial decision-making processes.")
st.subheader("About this model")
st.write("This is a simple beginner level machine learning project to predict only two outputs,or weather "
         "conditions that is whether it is a rainy day or sunny day."
         "the key features in the data sets are explained below")
st.write("MinTemp: Minimum temperature in degrees Celsius recorded for the day.")
st.write("MaxTemp: Maximum temperature in degrees Celsius recorded for the day.")
st.write("Evaporation: The evaporation measured in millimeters for the day.")
st.write("Sunshine: Number of hours of bright sunshine recorded during the day.")
st.write("WindGustDir: The direction of the strongest wind gust during the day.")
st.write("WindGustSpeed: Speed (in kilometers per hour) of the strongest wind gust during the day.")
st.write("WindDir9am: Wind direction at 9 am.")
st.write("WindDir3pm: Wind direction at 3 pm.")
st.write("WindSpeed9am: Wind speed (in kilometers per hour) at 9 am.")
st.write("WindSpeed3pm: Wind speed (in kilometers per hour) at 3 pm.")
st.write("Humidity9am: Relative humidity at 9 am as a percentage.")
st.write("Humidity3pm: Relative humidity at 3 pm as a percentage.")
st.write("Pressure3pm: Atmospheric pressure (in hPa) at 3 pm.")
st.write("Cloud9am: Fraction of sky obscured by cloud at 9 am.")
st.write("Cloud3pm: Fraction of sky obscured by cloud at 3 pm.")
st.write("Temp3pm: Temperature in degrees Celsius at 3 pm.")
st.write("RISK_MM: The amount of rain recorded for the day in millimeters.")








loaded_model = pickle.load(open("C:/Users/User/Pictures/machine learning project/trained_weather_model.sav", "rb"))

with open('C:/Users/User/Pictures/machine learning project/data.pkl', 'rb') as file:
    data = pickle.load(file)
st.write("please kindly use the encoded values for WindGustDir,WindDir9am,WindDir3pm while entring the input"
         " the encoded values are given below")
st.write("N=NORTH","S=SOUTH","E=EAST","W=WEST")

st.write("windgustdir:{E: 0, ENE: 1, ESE: 2, N: 3, NE: 4, NNE: 5, NNW: 6,"
                      "NW: 7, S: 8, SE: 9, SSE: 10, SSW: 11, SW: 12, W: 13, WNW: 14, WSW: 15}")
st.write("windDir9am:{E: 0, ENE: 1, ESE: 2, N: 3, NE: 4, NNE: 5, NNW: 6, "
                      "NW: 7, S: 8, SE: 9, SSE: 10, SSW: 11, SW: 12, W: 13, WNW: 14, WSW: 15}")
st.write("windDir3pm:{E: 0, ENE: 1, ESE: 2, N: 3, NE: 4, NNE: 5, NNW: 6,"
                     "NW: 7, S: 8, SE: 9, SSE: 10, SSW: 11, SW: 12, W: 13, WNW: 14, WSW: 15}")
st.dataframe(data.head())
def weather_prediction(input_data):
    input_data_numpy = np.asarray(input_data)
    input_data_reshape = input_data_numpy.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)

    if prediction == 1:
        return "IT'S A RAINY DAY!"

    else:
        return "IT'S A SUNNY DAY!"



def main():
    st.sidebar.header("User Input")
    MinTemp = st.sidebar.slider("Min Temperature", float(data["MinTemp"].min()), float(data["MinTemp"].max()),
                                float(data["MinTemp"].mean()))

    MaxTemp = st.sidebar.slider("Max Temperature", float(data["MaxTemp"].min()), float(data["MaxTemp"].max()),
                                float(data["MaxTemp"].mean()))

    Evaporation = st.sidebar.slider("Evaporation ", float(data["Evaporation"].min()), float(data["Evaporation"].max()),
                                    float(data["Evaporation"].mean()))

    Sunshine = st.sidebar.slider("Sunshine", float(data["Sunshine"].min()), float(data["Sunshine"].max()),
                                 float(data["Sunshine"].mean()))

    WindGustDir = st.sidebar.slider("WindGustDir", float(data["WindGustDir"].min()), float(data["WindGustDir"].max()),
                                    float(data["WindGustDir"].mean()))

    WindGustSpeed = st.sidebar.slider("WindGustSpeed", float(data["WindGustSpeed"].min()),
                                      float(data["WindGustSpeed"].max()), float(data["WindGustSpeed"].mean()))

    WindDir9am = st.sidebar.slider("WindDir9am", float(data["WindDir9am"].min()), float(data["WindDir9am"].max()),
                                   float(data["WindDir9am"].mean()))

    WindDir3pm = st.sidebar.slider("WindDir3pm", float(data["WindDir3pm"].min()), float(data["WindDir3pm"].max()),
                                   float(data["WindDir3pm"].mean()))

    WindSpeed9am = st.sidebar.slider("WindSpeed9am", float(data["WindSpeed9am"].min()),
                                     float(data["WindSpeed9am"].max()), float(data["WindSpeed9am"].mean()))

    WindSpeed3pm = st.sidebar.slider("WindSpeed3pm", float(data["WindSpeed3pm"].min()),
                                     float(data["WindSpeed3pm"].max()), float(data["WindSpeed3pm"].mean()))

    Humidity9am = st.sidebar.slider("Humidity9am", float(data["Humidity9am"].min()), float(data["Humidity9am"].max()),
                                    float(data["Humidity9am"].mean()))

    Humidity3pm = st.sidebar.slider("Humidity3pm", float(data["Humidity3pm"].min()), float(data["Humidity3pm"].max()),
                                    float(data["Humidity3pm"].mean()))

    Pressure3pm = st.sidebar.slider("Pressure3pm", float(data["Pressure3pm"].min()), float(data["Pressure3pm"].max()),
                                    float(data["Pressure3pm"].mean()))
    Cloud9am = st.sidebar.slider("Cloud9am", float(data["Cloud9am"].min()), float(data["Cloud9am"].max()),
                                 float(data["Cloud9am"].mean()))

    Cloud3pm = st.sidebar.slider("Cloud3pm", float(data["Cloud3pm"].min()), float(data["Cloud3pm"].max()),
                                 float(data["Cloud3pm"].mean()))

    Temp3pm = st.sidebar.slider("Temp3pm", float(data["Temp3pm"].min()), float(data["Temp3pm"].max()),
                                float(data["Temp3pm"].mean()))

    RISK_MM = st.sidebar.slider("RISK_MM", float(data["RISK_MM"].min()), float(data["RISK_MM"].max()),
                                float(data["RISK_MM"].mean()))
    # code for prediction
    # code for prediction
    prediction = ""
    if st.button("Weather Predict"):
        prediction = weather_prediction(
            [MinTemp, MaxTemp, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am,
             WindSpeed3pm, Humidity9am, Humidity3pm, Pressure3pm, Cloud9am, Cloud3pm, Temp3pm, RISK_MM])

    # Display the prediction result
    st.subheader("Prediction")
    st.write("The predicted result: ", prediction)
    st.success(prediction)
    from PIL import Image
    img=Image.open("C:/Users/User/PycharmProjects/pythonProject/package/mypack/rainyday.jpg")
    img2=Image.open("C:/Users/User/PycharmProjects/pythonProject/package/mypack/sunnyday.jpg")
    if prediction == "IT'S A RAINY DAY!":
        st.image(img)
    else:
        st.image(img2)
    # st.success(prediction)

    st.header("Data Visualization")

    # Line chart for MinTemp and MaxTemp
    st.subheader("Temperature Trend")
    st.write("the temperature trend of minimum and maximum temperatures are plotted below,and x shows the no of days")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=data.index, y=data["MinTemp"], label="Min Temp")
    sns.lineplot(x=data.index, y=data["MaxTemp"], label="Max Temp")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("MinTemp and MaxTemp Trend")
    st.pyplot()

    st.subheader("WindGustSpeed")
    st.write("The windspeed of each days are plotted in the graph")
    wind_speed_counts = data["WindGustSpeed"].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=wind_speed_counts.index, y=wind_speed_counts.values)
    plt.xlabel("Wind Gust speed ")
    plt.ylabel("Count")
    plt.title("Wind Gust speed Distribution")
    plt.xticks(rotation=45)
    st.pyplot()
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # Dynamic Plot Customization
    selected_variable = st.selectbox("Select Variable to Plot", ["Evaporation",	"Sunshine", "Humidity3pm"])
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=data.index, y=data[selected_variable], label=selected_variable)
    plt.xlabel("Date")
    plt.ylabel(selected_variable)
    plt.title(f"{selected_variable} Trend")
    st.pyplot()


if __name__ == "__main__":
    main()
 # streamlit run "C:\Users\User\Pictures\machine learning project\streamlitweatherpredict1.py"
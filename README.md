# Projet-Hackathon
Hackathon ENPC x eleven: Waiting times at an amusement park

This code aims to predict waiting times two hours in advance for the three main attractions of an amusement park: Flying Coaster, Pirate Ship, and Water Ride. 
We developed a predictive model that estimates the waiting times for these key attractions, based on provided data:

* a train set, "waiting_times_train.csv" with several features
  * date and hour (DATETIME),
  * name of attraction (ENTITY_DESCRIPTION_SHORT),
  * number of people the attraction can take (ADJUST_CAPACITY),
  * time during which the attraction is down (DOWNTIME),
  * current waiting time for the attraction (CURRENT_WAIT_TIME),
  * 3 features correspunding to "time until a special event in the park" (TIME_TO_PARADE_1, TIME_TO_PARADE_2, TIME_TO_NIGHT_SHOW),
  * and finally the feature we are training our model to predict (WAIT_TIME_IN_2H)

* a set "weather_data.csv" providing extra information on the weather:
  * temperature (temp)
  * dew_point
  * feels_like
  * pressure
  * humidity
  * wind_speed
  * rain_1h
  * snow_1h
  * clouds_all
  * DATETIME

We also have access to the validation set (waiting_times_X_test_val.csv) and test set (waiting_times_X_test_final.csv), though they are not used in this code (we test the model on a separate website for the Hackathon)

# HydroPower_System_Simulation
This git repository simulates a comprehensive hydropower system, including the influence of various factors on a single reservoir. It's designed as a scalable model, meaning its behavior adapts based on different input measurements. Future enhancements will allow it to simulate multiple interconnected reservoirs. In addition, the simulation will use Machine Learning methods in order to approximate optimial release patterns to maximize the energy output of the system. 

# Long Short Term Memory (LSTM) Model Description

The LSTM Model begins by importing the necessary libraries such as tensorFlow, Pandas, Sklearn and Numpy to name a few. It then ingest data from a premade CSV containing discharge total changes over time obtained from the International Boundary and Water Commission (IBWC). The data is then processed into a Pandas DataFrame to allow easy mutability, with timestamps converted to a datetime format and set as the index. Next we normalize the 'Value' column using a MinMaxScaler and create sequences of data for training. The current LSTM model is defined with two LSTM layers and one output Dense layer. This is subject to change as this model is only the first iteration. However, after compiling the model with the Adam optimizer and mean squared error loss, the data is split between training and testing sets for which the model is trained for 50 epochs (Also subject to change) and predicts based on the time series. Lastly, we evaluate the performance using mean squared error and mean absolute error which will later grow in differnt performance measures.


## How to run Simulator


### Install all dependencies and libraries.

1. Create a virtual environment to install all dependencies (optional but recommended)
```
python -m venv myenv
```

2. activate the virtual environment
```
myenv\Scripts\activate
```

3. In the terminal run:
```
pip install -r requirements.txt
```

### Run simulator

1. Once the virtual environment is activated, run the following in the temrinal:
```
python Simulator.py
```
Then go to http://127.0.0.1:5000

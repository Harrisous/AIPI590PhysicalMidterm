# AIPI590PhysicalMidterm
## Background/introduction of the idea
This project is a car device built for cultivating save driving behavior. It has 2 modes: driving mode and reverse mode. 

- When in driving mode, the device will collect the acceleration and gyroscope reading and predict the driver's behavior from a trained model, and give the warning via LED light when driving aggressively. The current driving behavior will be displayed on the LCD screen.
- In reverse mode, the distance sensor will measure the distance and give warning via LED light when close to an obstacle. The user can set the safty distance boundard by a potentiometer. Both current distance and "danger close" will be displayed on the LCD screen when in this mode. 
- The mode switch is done by a button, which is a simulation of the switch of the gear.

## Sensors, inputs and peripherals used
Link to cirkitdesigner: https://app.cirkitdesigner.com/project/89b9f252-3b6f-41fc-8c1b-38e9deaafd63
- LCD screen 16*2 I2C
- LED Green & Red
- Resistor: 220Ω * 2, 1kΩ * 1, 2.2kΩ * 1
- HC-SR04 Ultrasonic Sensor
- ADS7830 ADC
- Potentiometer
- **InvenSense MPU6050**

## What model is predicting
Given acceleration and gyroscope readings, the random forest model is trained to predict the category of driving behavior. There are 3 classes: NORMAL, AGGRESSIVE, SLOW.

## Challenges came across
- Idea for the project.
- Data collection. It is solved by looking into Kaggle data set (which is listed in the next section).
- Quantize Random Forest Model. Torch does not support random forest training since it is built for deep learning tasks. But after consulting with Perplexity, I chose onnx to store and quantize the model.
- Gyroscope readings. This is done with the hardware knowledge from Perplexity.

## Data source
https://www.kaggle.com/datasets/outofskills/driving-behavior?select=train_motion_data.csv

## A quick explanation of how it works
Script will first load and process data from the folder, and do preprocessing. Then a grid search will be done with 3 cross validation and AUROC as the metric. The best model will be picked and retrained on the dataset. Finally, the model is quantized and saved as an onnx file. After cloning the model to raspberry pi, the other script will extract the model, get the readings from MPU6050 and predict the class. The class is then mapped back to behavior type and it will be displayed on LCD screen. That is all about the main functionality.

## How it is trained
The model is first trained on the remote computer/laptop:
0. clone the repo `git clone https://github.com/Harrisous/AIPI590PhysicalMidterm.git` (when it is public)
1. run `python -m venv .venv`
2. activate .venv `.venv\Scripts\activate`(windows cmd)
3. install dependencies: `pip install -r requirements_train.txt`
4. run `python train.py` and the script will generate the model and export it to onnx format with original size and compressed size.
Then the model can be applied to RaspberryPi:
0. repeat the steps 0-2 from the previous steps
1. install dependencies: `pip install -r requirements_run.txt` this is all the libraries from my raspberry pi.
2. run `python main.py` to activate the system.

## A demonstration of how it works
- Demo video link: 
- Presentation deck:

## Next steps
1. Train deep learn models that can be easier handled by torch and make the prediction perform better. (Current AUROC ovr is only about 0.6)
2. Build the system on to a car and test its reliability. Toy car may apply.
3. Add voice control or other models and build it into a bigger smart device on car.
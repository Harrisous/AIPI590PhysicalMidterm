'''This is the script to run on RaspberryPi'''
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor
from lcd_i2c import LCD_I2C
from gpiozero import Button
import time
import smbus
import onnxruntime as ort
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="For more accurate readings, use the pigpio pin factory")

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class LcdDevice:
    '''
        <LCD screen> display current speed, in percentage
    '''
    def __init__(self, DEVICE_ADDRESS = 0x27, width=16, length=2):
        lcd = LCD_I2C(DEVICE_ADDRESS, width, length)
        lcd.backlight.on()
        lcd.blink.off()
        self.lcd = lcd

    def update_lcd(self, line1="", line2=""):
        '''Update the speed number with new distance and new speed'''
        # update display
        self.lcd.clear()
        self.lcd.cursor.setPos(0, 4)
        self.lcd.write_text(f"{line1}")
        self.lcd.cursor.setPos(1, 4)
        self.lcd.write_text(f"{line2}")

    def cleanup(self):
        self.lcd.backlight.off()
        self.lcd.clear()


class Ultrasonic:
    '''
        Class for representing Ultrasonic sensor
    '''
    def __init__(self, TRIGGER_PIN=24, ECHO_PIN=23):
        self.ultrasonic = DistanceSensor(echo = ECHO_PIN,trigger = TRIGGER_PIN)
    
    def measure_distance(self):
        distance_cm = self.ultrasonic.distance * 100
        return distance_cm
    

class Led:
    '''Class to represent LEDs'''
    def __init__(self, LED_PIN):
        self.LED_PIN = LED_PIN
        GPIO.setup(self.LED_PIN, GPIO.OUT)
    
    def turn_on(self):
        GPIO.output(self.LED_PIN, GPIO.HIGH)

    def turn_off(self):
        GPIO.output(self.LED_PIN, GPIO.LOW)


class Potentiometer:
    def __init__(self, DEVICE_ADDRESS=0x4b, channel=0):
        self.I2C_BUS = 1
        self.DEVICE_ADDRESS = DEVICE_ADDRESS
        self.bus = smbus.SMBus(self.I2C_BUS)
        self.channel = channel
        self.control_byte = 0x84 | (self.channel << 4)
    
    def reading(self):
        self.bus.write_byte(self.DEVICE_ADDRESS, self.control_byte)
        reading = self.bus.read_byte(self.DEVICE_ADDRESS)
        reading_scale100 = round(reading/255*100)
        return reading_scale100 


class Gyroscope: 
    '''Using MPU6050, this class is finished with the help of Perplexity'''
    def __init__(self, DEVICE_ADDRESS=0x68):
        self.bus = smbus.SMBus(1)  # Use I2C bus 1
        self.address = DEVICE_ADDRESS
        self.bus.write_byte_data(self.address, 0x6B, 0)  # Wake up the MPU6050 from sleep mode
        self.accel_bias = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gyro_bias = {"x": 0.0, "y": 0.0, "z": 0.0}
    
    def read_raw_data(self, reg):
        '''
        Read raw data from a given register.
        '''
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        if value > 32768:  # Convert to signed value
            value -= 65536
        return value

    def get_data(self, num_samples=50):
        '''
        Get accelerometer data for X, Y, Z axes in g (gravity).
        Get gyroscope data for X, Y, Z axes in degrees/second.
        '''
        #
        acc_x_sum = acc_y_sum = acc_z_sum = gyro_x_sum = gyro_y_sum = gyro_z_sum = 0.0
        
        for _ in range(num_samples):
            acc_x = self.read_raw_data(0x3B) / 16384.0  # Scale factor for ±2g range
            acc_y = self.read_raw_data(0x3D) / 16384.0
            acc_z = self.read_raw_data(0x3F) / 16384.0
            gyro_x = self.read_raw_data(0x43) / 131.0  # Scale factor for ±250°/s range
            gyro_y = self.read_raw_data(0x45) / 131.0
            gyro_z = self.read_raw_data(0x47) / 131.0
            
            acc_x_sum += acc_x
            acc_y_sum += acc_y
            acc_z_sum += acc_z
            gyro_x_sum += gyro_x
            gyro_y_sum += gyro_y
            gyro_z_sum += gyro_z
            time.sleep(0.01)
        
        acc_x = acc_x_sum / num_samples
        acc_y = acc_y_sum / num_samples
        acc_z = acc_z_sum / num_samples
        gyro_x = gyro_x_sum / num_samples
        gyro_y = gyro_y_sum / num_samples
        gyro_z = gyro_z_sum / num_samples 

        acc_x -= self.accel_bias["x"]
        acc_y -= self.accel_bias["y"]
        acc_z -= self.accel_bias["z"]
        gyro_x -= self.gyro_bias["x"]
        gyro_y -= self.gyro_bias["y"]
        gyro_z -= self.gyro_bias["z"]

        return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

    def calibrate(self, num_samples=100):
        '''
        Calibrate the gyroscope by calculating bias (offset) when stationary.
        '''
        sum_x_a = sum_y_a = sum_z_a = 0.0
        sum_x = sum_y = sum_z = 0.0
        
        for _ in range(num_samples):
            acc_x = self.read_raw_data(0x3B) / 16384.0 
            acc_y = self.read_raw_data(0x3D) / 16384.0
            acc_z = self.read_raw_data(0x3F) / 16384.0
            gyro_x = self.read_raw_data(0x43) / 131.0
            gyro_y = self.read_raw_data(0x45) / 131.0
            gyro_z = self.read_raw_data(0x47) / 131.0
            
            sum_x_a += acc_x
            sum_y_a += acc_y
            sum_z_a += acc_z
            sum_x += gyro_x
            sum_y += gyro_y
            sum_z += gyro_z
            
            time.sleep(0.01)
        self.gyro_bias["x"] = sum_x / num_samples
        self.gyro_bias["y"] = sum_y / num_samples
        self.gyro_bias["z"] = sum_z / num_samples
        
        print(f"Calibration complete!")


class BehaviorPredictor:
    '''
    Class to load an ONNX model and predict behavior using accelerometer and gyroscope data.
    This class is generated by Perplexity but modified based on my need.
    '''
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        # Get input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        # prepare input data as a NumPy array
        # Run inference
        predictions = self.session.run([self.output_name], {self.input_name: input_data})
        predicted_class = predictions[0][0] 

        return predicted_class


if __name__ == "__main__":
    try:
        # initialize car state
        driving_forward = True

        # instantiate devices
        lcd = LcdDevice(DEVICE_ADDRESS = 0x27)
        ultrasonic = Ultrasonic(TRIGGER_PIN=24, ECHO_PIN=23)
        btn = Button(16, pull_up=True, bounce_time=0.05)
        def change_state():
            global driving_forward
            driving_forward = not driving_forward
        btn.when_pressed = change_state # GPIO: RuntimeError: Failed to add edge detection
        led_green = Led(LED_PIN=21)
        led_red = Led(LED_PIN=20)
        led_green.turn_off()
        led_red.turn_off()
        potent = Potentiometer(DEVICE_ADDRESS=0x4b, channel=0)
        gyro = Gyroscope(DEVICE_ADDRESS=0x68)
        gyro.calibrate()
        # read model for further prediction
        model_path = "core_model_INT8.onnx"
        predictor = BehaviorPredictor(model_path)
        # read fitted scaler to scale input
        scaler = joblib.load('scaler.save')
        result_mapping = {2:"SLOW",1:"NORMAL",0:"Aggressive"}
        # enter loop
        while True:
            if driving_forward:
                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = gyro.get_data()
                sensor_data = np.array([[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]],dtype=np.float32)
                scaled_data = scaler.transform(sensor_data)# scale input
                predicted_behavior = predictor.predict(scaled_data) # make prediction
                # display
                lcd.update_lcd(line1="Driving is", line2=f"{result_mapping[predicted_behavior]}")
                if predicted_behavior == 0:
                    led_green.turn_off()
                    led_red.turn_on()
                else:
                    led_green.turn_on()
                    led_red.turn_off()
            else:
                distance = int(ultrasonic.measure_distance())
                sensitivity = potent.reading()
                if distance >= sensitivity:
                    led_green.turn_on()
                    led_red.turn_off()
                else:
                    led_green.turn_off()
                    led_red.turn_on()
                lcd.update_lcd(line1=f"{distance}cm",line2=f"sensi:{sensitivity}")
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n Exiting Program")
    except Exception as e:
        print(e)
    finally:
        lcd.cleanup()
        GPIO.cleanup()




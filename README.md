# MantaPos

> This project uses **Python 3.11.9** and **Arduino IDE 2.3.3**.

Make sure to install the required packages:

```sh
pip install opencv-contrib-python
```

```sh
pip install opencv-python pytesseract   
```
Also install Tesseract OCR via the [Tesseract GitHub Wiki](https://github.com/UB-Mannheim/tesseract/wiki).

## Arduino Libraries (for ESP32-PoE-ISO)
From IDE Library Manager: 
- Adafruit BusIO
- Adafruit ADS1X15

From GitHub:
> Place the folder containing the main "library_name.h" file in <br /> 
> `C:\Users\<USER>\Documents\Arduino\libraries`<br /> 
> and restart the Arduino IDE.
- [ESP32-PoE](https://github.com/OLIMEX/ESP32-POE/tree/master/SOFTWARE/ARDUINO)
- [DS3231](https://github.com/OLIMEX/MOD-RTC2/tree/master/Software/MOD-RTC2) *- Rename folder to DS3231 first*
- [I2Cdev and MPU6050](https://github.com/jrowberg/i2cdevlib/tree/master/Arduino)
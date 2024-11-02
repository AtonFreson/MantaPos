/*
  DS3231: Real-Time Clock. Temperature example
  Modified for ESP32
*/

#include <Wire.h>
#include <DS3231.h>

// ESP32 I2C pins
#define I2C_SDA 13  // Default SDA pin for ESP32-POE-ISO
#define I2C_SCL 16  // Default SCL pin for ESP32-POE-ISO

DS3231 RTC_clock;
RTCDateTime dt;

void scanI2C() {
  byte error, address;
  int nDevices = 0;
 
  Serial.println("Scanning I2C bus...");
 
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
 
    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.println(address, HEX);
      nDevices++;
    }
  }
  
  if (nDevices == 0) {
    Serial.println("No I2C devices found");
  }
}

void setup()
{
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  
  // Scan I2C bus first
  scanI2C();
  
  // Initialize DS3231
  Serial.println("Initialize DS3231");
  RTC_clock.begin();
  
  delay(1000); // Give time for RTC to stabilize
}

void loop()
{
  // Force temperature conversion
  RTC_clock.forceConversion();

  float temp = RTC_clock.readTemperature();
  
  Serial.print("Temperature: ");
  Serial.println(temp);

  delay(1000);
}

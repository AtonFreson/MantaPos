#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <I2Cdev.h>
#include <MPU6050.h>
#include <Wire.h>
#include <DS3231.h>
#include <Adafruit_ADS1X15.h>

// Pin definitions for SCA50 differential signals
#define ENCODER_PIN_A_POS 14  // A
#define ENCODER_PIN_A_NEG 15  // A-
#define ENCODER_PIN_B_POS 32  // B
#define ENCODER_PIN_B_NEG 33  // B-
#define INDEX_PIN_Z_POS 35    // Z
#define INDEX_PIN_Z_NEG 36    // Z-

// ESP32-PoE-ISO I2C pins
#define I2C_SDA 13
#define I2C_SCL 16
#define I2C_1_SDA 5
#define I2C_1_SCL 4
 
// Encoder configuration
#define PPR 400  // Pulses Per Revolution
#define COUNTS_PER_REV (PPR * 4)  // In quadrature mode
#define WHEEL_DIAMETER_MM 82.53
#define DISTANCE_PER_REV (WHEEL_DIAMETER_MM * PI / 1000.0)  // meters per revolution
#define DISTANCE_PER_COUNT (DISTANCE_PER_REV / COUNTS_PER_REV)

volatile long encoderCount = 0;
long lastEncoderCount = 0;
unsigned long lastPrintTime = 0;
unsigned long lastMeasurementTime = 0;
unsigned long encoderTimestamp = millis();
const unsigned long printInterval = 100; // 10ms update rate
volatile bool resetCommandReceived = false;
volatile bool resetOccurred = false;
const bool debugLock = false; // Set to true to ensure serial connection;
// Otherwise just send anything to device when connected and serial output will enable itself.

// Network configuration
IPAddress local_ip(169, 254, 178, 100);    // ESP32 static IP (adjusted to match PC's subnet)
IPAddress gateway(0, 0, 0, 0);             // No gateway needed for direct connection
IPAddress subnet(255, 255, 0, 0);          // Subnet mask matching PC's subnet mask
IPAddress dns(0, 0, 0, 0);                 // No DNS server needed

// UDP configuration
const char* udpAddress = "169.254.255.255"; // Broadcast address or specific IP
const int udpPort = 13233;                // Source & Destination port

WiFiUDP udp;

// IMU configuration
MPU6050 accelgyro(0x69); // Make sure AD0 is high by shorting R3 to VCC

// Variables to hold data
int16_t ax, ay, az;
int16_t gx, gy, gz;
long currentCount;
float revolutions = 0, rpm = 0, speed = 0, distance = 0;
unsigned long imuTimestamp, tempTimestamp, pressureTimestamp;
float ax_mss, ay_mss, az_mss;
float gx_rads, gy_rads, gz_rads;
float temperature;
int16_t adc_value;
float g = 9.80665; // m/s²

// Temperature sensor configuration
DS3231 temp_sensor;

// ADC configuration for pressure sensor
Adafruit_ADS1115 ads;  /* 16-bit version */

// Wrapper class for Serial to handle connection status
class SerialWrapper {
public:
    void begin(unsigned long baudrate) {
        Serial.begin(baudrate);
        this->serialConnected = true;
        
        // Define timeout period (e.g., 2000 milliseconds)
        unsigned long timeout = 2000;
        unsigned long startTime = millis();
        
        // Wait for serial connection
        while (!Serial && (millis() - startTime) < timeout) {
            // Do nothing, just wait
        }
    }
    
    template<typename T>
    void println(T message) {
        if (this->serialConnected) {
            Serial.println(message);
        }
    }

    template<typename T>
    void print(T message) {
        if (this->serialConnected) {
            Serial.print(message);
        }
    }
    
    int available() {
        return Serial.available();
    }
    
    String readString() {
        return Serial.readString();
    }
    
    bool getSerialConnected() {
        return this->serialConnected;
    }
    
    void setSerialConnected(bool status) {
        this->serialConnected = status;
    }

private:
    bool serialConnected = false;
};

SerialWrapper SerialW;

// Differential signal processing
inline int readDifferential(int pinPos, int pinNeg) {
    return digitalRead(pinPos) - digitalRead(pinNeg);
}

void IRAM_ATTR encoderISR() {
    static int8_t lookup_table[] = {0,-1,1,0,1,0,0,-1,-1,0,0,1,0,1,-1,0};
    static uint8_t lastState = 0;
    
    encoderTimestamp = millis();

    // Read current state of both channels using differential signals
    int8_t stateA = readDifferential(ENCODER_PIN_A_POS, ENCODER_PIN_A_NEG);
    int8_t stateB = readDifferential(ENCODER_PIN_B_POS, ENCODER_PIN_B_NEG);
    
    // Convert to absolute states (0 or 1)
    stateA = stateA > 0 ? 1 : 0;
    stateB = stateB > 0 ? 1 : 0;
    
    // Create current state from both channels
    uint8_t currentState = (stateA << 1) | stateB;
    
    // Create lookup index from last and current state
    uint8_t index = (lastState << 2) | currentState;
    
    // Update count based on state transition
    encoderCount += lookup_table[index];
    
    // Save current state for next iteration
    lastState = currentState;
}

void IRAM_ATTR indexISR() {
    if (readDifferential(INDEX_PIN_Z_POS, INDEX_PIN_Z_NEG) > 0) {
        if (resetCommandReceived) {
            lastEncoderCount = lastEncoderCount - encoderCount; // Adjust the last count relative to zero
            encoderCount = 0; // Reset the encoder count
            resetCommandReceived = false; // Clear the reset command flag
            resetOccurred = true; // Set the reset occurred flag
        }
    }
}

void WiFiEvent(arduino_event_t *event) {
    switch (event->event_id) {
        case ARDUINO_EVENT_ETH_START:
            SerialW.println("Ethernet Started");
            // Set the hostname for the ESP32
            ETH.setHostname("esp32-ethernet");
            break;
        case ARDUINO_EVENT_ETH_CONNECTED:
            SerialW.println("Ethernet Connected");
            break;
        case ARDUINO_EVENT_ETH_GOT_IP:
            SerialW.println("Ethernet Got IP");
            SerialW.print("IP Address: ");
            SerialW.println(ETH.localIP());
            break;
        case ARDUINO_EVENT_ETH_DISCONNECTED:
            SerialW.println("Ethernet Disconnected");
            break;
        case ARDUINO_EVENT_ETH_STOP:
            SerialW.println("Ethernet Stopped");
            break;
        default:
            break;
    }
}


void dataPrint() {
    // Encoder data capture
    noInterrupts();  // Disable interrupts while reading volatile
    currentCount = encoderCount;
    interrupts();

    // Calculate time delta since last measurement
    unsigned long timeDelta = encoderTimestamp - lastMeasurementTime;
    if (timeDelta > 0) {
        revolutions = (float)currentCount / COUNTS_PER_REV;
        distance = currentCount * DISTANCE_PER_COUNT;

        // Calculate speed
        float countsPerSec = (float)(currentCount - lastEncoderCount) * (1000.0 / timeDelta);
        rpm = (countsPerSec * 60.0) / COUNTS_PER_REV;
        speed = countsPerSec * DISTANCE_PER_COUNT;

        lastEncoderCount = currentCount;
        lastMeasurementTime = encoderTimestamp;
    }

    // IMU data capture
    imuTimestamp = millis();
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw acceleration data to m/s², and raw gyroscope data to radians per second
    ax_mss = (float)ax / 16384.0 * g;
    ay_mss = (float)ay / 16384.0 * g;
    az_mss = (float)az / 16384.0 * g;
    gx_rads = (float)gx / 131.0 * (PI / 180.0);
    gy_rads = (float)gy / 131.0 * (PI / 180.0);
    gz_rads = (float)gz / 131.0 * (PI / 180.0);

    // Temperature data capture
    tempTimestamp = millis();
    temp_sensor.forceConversion();
    temperature = temp_sensor.readTemperature();

    // Pressure data capture
    pressureTimestamp = millis();
    adc_value = ads.readADC_SingleEnded(0);

    // Print data to Serial if connected
    if (SerialW.getSerialConnected()) {
        Serial.println("Encoder Data (T:" + String(encoderTimestamp) + "):");
        Serial.print("Rev: "); Serial.print(revolutions,5); Serial.print("\t");
        Serial.print("RPM: "); Serial.print(rpm,5); Serial.print("\t");
        Serial.print("Speed (m/s): "); Serial.print(speed,5); Serial.print("\t");
        Serial.print("Distance (m): "); Serial.print(distance,5); Serial.println();
        Serial.println();

        Serial.println("IMU Data (T:" + String(imuTimestamp) + "):");
        Serial.print("Acceleration (m/s²): ");
        Serial.print("X="); Serial.print(ax_mss,5); Serial.print("\t");
        Serial.print("Y="); Serial.print(ay_mss,5); Serial.print("\t");
        Serial.print("Z="); Serial.print(az_mss,5); Serial.println();
        Serial.print("Gyroscope (rad/s): ");
        Serial.print("X="); Serial.print(gx_rads,5); Serial.print("\t");
        Serial.print("Y="); Serial.print(gy_rads,5); Serial.print("\t");
        Serial.print("Z="); Serial.print(gz_rads,5); Serial.println();
        Serial.println();

        Serial.println("Temperature Data (T:" + String(tempTimestamp) + "):");
        Serial.print("Temperature: "); Serial.println(temperature);
        Serial.println();

        Serial.println("Pressure Data (T:" + String(pressureTimestamp) + "):");
        Serial.print("Pressure: "); Serial.println(adc_value);
        Serial.println();
    }

    // Assemble JSON string for UDP
    char jsonBuffer[512];
    snprintf(jsonBuffer, sizeof(jsonBuffer),
      "{\"encoder\": {\"timestamp\": %lu, \"revolutions\": %.3f, \"rpm\": %.3f, \"speed\": %.3f, \"distance\": %.3f}, "
      "\"imu\": {\"timestamp\": %lu, \"acceleration\": {\"x\": %.5f, \"y\": %.5f, \"z\": %.5f}, \"gyroscope\": {\"x\": %.5f, \"y\": %.5f, \"z\": %.5f}}, "
      "\"temperature\": {\"timestamp\": %lu, \"value\": %.2f}, "
      "\"pressure\": {\"timestamp\": %lu, \"adc_value\": %d}}",
      encoderTimestamp, revolutions, rpm, speed, distance,
      imuTimestamp, ax_mss, ay_mss, az_mss, gx_rads, gy_rads, gz_rads,
      tempTimestamp, temperature,
      pressureTimestamp, adc_value);

    // Send over UDP
    udp.beginPacket(udpAddress, udpPort);
    udp.print(jsonBuffer);
    udp.endPacket();
}


void setup() {
    delay(250);
    SerialW.begin(115200);

    // Configure all pins as inputs
    pinMode(ENCODER_PIN_A_POS, INPUT);
    pinMode(ENCODER_PIN_A_NEG, INPUT);
    pinMode(ENCODER_PIN_B_POS, INPUT);
    pinMode(ENCODER_PIN_B_NEG, INPUT);
    pinMode(INDEX_PIN_Z_POS, INPUT);
    pinMode(INDEX_PIN_Z_NEG, INPUT);
    
    // Attach interrupts - using CHANGE for both edges of the signals, and RISING for index
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A_POS), encoderISR, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_B_POS), encoderISR, CHANGE);
    attachInterrupt(digitalPinToInterrupt(INDEX_PIN_Z_POS), indexISR, RISING);
    
    delay(1000); // Wait a bit for serial connection to be established
    SerialW.println("\nSCA50-400 Encoder Test");
    SerialW.println("PPR: 400");
    SerialW.println("Resolution: ±0.10mm\n");

    // Initialize Ethernet and register event handler
    WiFi.onEvent(WiFiEvent);
    ETH.begin();

    // Optional: Set static IP configuration
    ETH.config(local_ip, gateway, subnet, dns);

    // Wait for Ethernet connection
    while (!ETH.linkUp()) {
        SerialW.println("Waiting for Ethernet connection...");
        delay(1000);
    }

    udp.begin(udpPort);  // Set the source port to 13233

    // Initialize IMU (accelerometer and gyroscope)
    Wire.begin(I2C_SDA, I2C_SCL);
    SerialW.println("Initializing IMU devices...");
    accelgyro.initialize();

    // Verify connection
    SerialW.println("Testing IMU connections...");
    if (accelgyro.testConnection()) {
      	SerialW.println("MPU6050 connection successful.");
    } else {
      	SerialW.println("MPU6050 connection failed, halting program.");
      	while (1); // halt the program
    }

    // Initialize DS3231
    SerialW.println("Initialize DS3231 - temperature sensor");
    temp_sensor.begin();
    
    // Initialize ADC
    Wire1.begin(I2C_1_SDA, I2C_1_SCL);
    SerialW.println("Initializing pressure sensor ADC...");
    if (!ads.begin(ADS1X15_ADDRESS, &Wire1)) {
        SerialW.println("Failed to initialize pressure sensor ADC, halting program.");
        while (1);
    }


    // Disable debug serial output if not in debug mode
    if (!debugLock) {
        SerialW.println("Turning off debug serial output, send 'debug on' to enable");
        SerialW.setSerialConnected(false);
    }
}

void loop() {
    // Check if a reset occurred and print the message
    if (resetOccurred) {
        SerialW.println("Encoder count reset.");
        resetOccurred = false; // Clear the reset occurred flag
    }

    unsigned long currentTime = millis();

    if (currentTime - lastPrintTime >= printInterval) {
        dataPrint();
        lastPrintTime = currentTime;
    }
    
    // Check for SerialW input
    if (SerialW.available() > 0) {
        String input = SerialW.readString();
        input.trim(); // Remove any leading/trailing whitespace
        SerialW.println("Received input: " + input);

        if (input.equals("debug on")) {
            SerialW.setSerialConnected(true);
            SerialW.println("Received input: " + input);
            SerialW.println("Turning on debug serial output");
        }

        if (input.equals("reset")) {
            SerialW.println("Reset upon next index pulse");
            noInterrupts();
            resetCommandReceived = true;
            interrupts();

        } else if (input.equals("debug off")) {
            SerialW.println("Turning off debug serial output");
            SerialW.setSerialConnected(false);
        }
    }
}
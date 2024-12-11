#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <I2Cdev.h>
#include <MPU6050.h>
#include <Wire.h>
#include <DS3231.h>
#include <Adafruit_ADS1X15.h>
#include <ArduinoJson.h>
#include <ESP1588.h>
#include <EEPROM.h>

// The ESP32 unit number. Following 0:"X-axis Encoder", 1:"Z-axis Encoder - Main", 2:"Z-axis Encoder - Second", 3:"Surface Pressure Sensor" 
#define MPU_UNIT 1

// Features available on this unit
#define HAS_CLOCK       1
#define HAS_ENCODER     1
#define HAS_IMU         0
#define HAS_PRESSURE    0

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

// RTC Interrupt Configuration
#define RTC_ISR_PIN 2
byte itr_freq = 0x00;

// Encoder configuration
#define PPR 400  // Pulses Per Revolution
#define COUNTS_PER_REV (PPR * 4)  // In quadrature mode
#define WHEEL_DIAMETER_MM 82.53
#define DISTANCE_PER_REV (WHEEL_DIAMETER_MM * PI / 1000.0)  // meters per revolution
#define DISTANCE_PER_COUNT (DISTANCE_PER_REV / COUNTS_PER_REV)

#if HAS_ENCODER
volatile long encoderCount = 0;
long lastEncoderCount = 0;
#endif

unsigned long lastPrintTime = 0;
unsigned long lastMeasurementTime = 0;
const unsigned long printInterval = 100; // in ms
volatile bool resetCommandReceived = false;
volatile bool resetOccurred = false;
const bool debugLock = false; // Set to true to ensure serial connection;
// Otherwise just send anything to device when connected and serial output will enable itself.

// Network configuration
IPAddress local_ip(169, 254, 178, 100 + MPU_UNIT);    // ESP32 static IP (needs to match PC's subnet)
IPAddress gateway(0, 0, 0, 0);             // No gateway needed for direct connection
IPAddress subnet(255, 255, 0, 0);          // Subnet mask matching PC's subnet mask
IPAddress dns(0, 0, 0, 0);                 // No DNS server needed

// UDP configuration
const char* udpAddress = "169.254.255.255"; // Broadcast address or specific IP
const int udpPort = 13233;                // Source & Destination port

// UDP instance for sending data
WiFiUDP udp;

// UDP instance for receiving commands
const int udpCommandPort = 13234;
WiFiUDP udpCommand;  // Added for receiving commands

// IMU configuration
#if HAS_IMU
MPU6050 accelgyro(0x69); // Make sure AD0 is high by shorting R3 to VCC
#endif

// Variables to hold data
#if HAS_IMU
int16_t ax, ay, az;
int16_t gx, gy, gz;
float ax_mss, ay_mss, az_mss;
float gx_rads, gy_rads, gz_rads;
#endif

#if HAS_ENCODER
long currentCount;
float revolutions = 0, rpm = 0, speed = 0, distance = 0;
#endif

#if HAS_PRESSURE
int16_t adc_value0;
int16_t adc_value1;
#endif

uint64_t imuTimestamp, tempTimestamp, pressureTimestamp;
uint64_t encoderTimestamp = millis();
uint64_t shiftNs = esp_timer_get_time() * 1000ULL;
volatile uint64_t localSecondOffsetNs = esp_timer_get_time() * 1000ULL;

float temperature;
float g = 9.818584897; // m/s²

// Real Time Clock configuration
#if HAS_CLOCK
DS3231 rtc;
#endif

const unsigned int localEventPort = 319;
const unsigned int localGeneralPort = 320;
WiFiUDP UdpEvent;
WiFiUDP UdpGeneral;

IPAddress ptpMulticastIP(224, 0, 1, 129);
IPAddress masterIP(169, 254, 178, 87);

int lastPrint = 0;
bool sync_clock = false;
uint32_t storedSyncTime = 0; // Store the last sync time read from EEPROM
int64_t storedOffset = 0;    // Variable to store the offset read from EEPROM

// ADC configuration for pressure sensor
#if HAS_PRESSURE
Adafruit_ADS1115 ads;  /* 16-bit version */
#endif

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
    
    template<typename... Args>
    void printf(const char* format, Args... args) {
        if (this->serialConnected) {
            Serial.printf(format, args...);
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

uint64_t getEpochMillis64(bool difference = false) {
    // If there is a valid PTP lock, return that time
    uint64_t millis_comp = 0;

    ESP1588_Tracker& m = esp1588.GetMaster();
    if (esp1588.GetLockStatus() && m.Healthy() && abs(esp1588.GetLastDiffMs()) <= 100) {
        millis_comp = esp1588.GetEpochMillis64();
        if (!difference) {
            return millis_comp;
        }
    }

#if HAS_CLOCK
    DateTime now = RTClib::now();
    noInterrupts();
    shiftNs = esp_timer_get_time() * 1000ULL - localSecondOffsetNs;
    interrupts();

    if (difference) {
        return 500000000ULL + (int64_t)(millis_comp - (((uint64_t)now.unixtime() * 1000ULL) + shiftNs / 1000000ULL)) + storedOffset / 1000000ULL;
    }
    return (((uint64_t)now.unixtime() * 1000ULL) + shiftNs / 1000000ULL + storedOffset / 1000000ULL);
#else
    // If there is no clock, return ESP timer
    return esp_timer_get_time() / 1000ULL;
#endif
}

#if HAS_ENCODER
// Differential signal processing
inline int readDifferential(int pinPos, int pinNeg) {
    return digitalRead(pinPos) - digitalRead(pinNeg);
}

void IRAM_ATTR encoderISR() {
    static int8_t lookup_table[] = { 0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0 };
    static uint8_t lastState = 0;

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
#endif

#if HAS_CLOCK
void rtcSecondInterrupt() {
    localSecondOffsetNs = esp_timer_get_time() * 1000ULL;
}
#endif

void WiFiEvent(arduino_event_t* event) {
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

void dataPrint(ESP1588_Tracker& m) {
#if HAS_ENCODER
    // Encoder data capture
    encoderTimestamp = getEpochMillis64();
    noInterrupts();  // Disable interrupts while reading volatile
    currentCount = encoderCount;
    interrupts();

    // Calculate distance
    revolutions = (float)currentCount / COUNTS_PER_REV;
    distance = currentCount * DISTANCE_PER_COUNT;

    // Calculate speed
    unsigned long timeDelta = encoderTimestamp - lastMeasurementTime;
    float countsPerSec = (float)(currentCount - lastEncoderCount) * (1000.0 / timeDelta);
    rpm = (countsPerSec * 60.0) / COUNTS_PER_REV;
    speed = countsPerSec * DISTANCE_PER_COUNT;

    lastEncoderCount = currentCount;
    lastMeasurementTime = encoderTimestamp;
#endif

#if HAS_IMU
    // IMU data capture
    imuTimestamp = getEpochMillis64();
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw acceleration data to m/s², and raw gyroscope data to radians per second
    ax_mss = (float)ax / 16384.0 * g;
    ay_mss = (float)ay / 16384.0 * g;
    az_mss = (float)az / 16384.0 * g;
    gx_rads = (float)gx / 131.0 * (PI / 180.0);
    gy_rads = (float)gy / 131.0 * (PI / 180.0);
    gz_rads = (float)gz / 131.0 * (PI / 180.0);
#endif

#if HAS_CLOCK
    // Temperature data capture
    tempTimestamp = getEpochMillis64();
    temperature = rtc.getTemperature();
#endif

#if HAS_PRESSURE
    // Pressure data capture
    pressureTimestamp = getEpochMillis64();
    adc_value0 = ads.readADC_SingleEnded(0);
    adc_value1 = ads.readADC_SingleEnded(1);
#endif

    // Serial output
    if (SerialW.getSerialConnected()) {
#if HAS_ENCODER
        // Encoder data printing
        Serial.println("Encoder Data (T:" + String(encoderTimestamp) + "):");
        Serial.print("Rev: "); Serial.print(revolutions, 5); Serial.print("\t");
        Serial.print("RPM: "); Serial.print(rpm, 5); Serial.print("\t");
        Serial.print("Speed (m/s): "); Serial.print(speed, 5); Serial.print("\t");
        Serial.print("Distance (m): "); Serial.print(distance, 5); Serial.println();
        Serial.println();
#endif

#if HAS_IMU
        // IMU data printing
        Serial.println("IMU Data (T:" + String(imuTimestamp) + "):");
        Serial.print("Acceleration (m/s²): ");
        Serial.print("X="); Serial.print(ax_mss, 5); Serial.print("\t");
        Serial.print("Y="); Serial.print(ay_mss, 5); Serial.print("\t");
        Serial.print("Z="); Serial.print(az_mss, 5); Serial.println();
        Serial.print("Gyroscope (rad/s): ");
        Serial.print("X="); Serial.print(gx_rads, 5); Serial.print("\t");
        Serial.print("Y="); Serial.print(gy_rads, 5); Serial.print("\t");
        Serial.print("Z="); Serial.print(gz_rads, 5); Serial.println();
        Serial.println();
#endif

#if HAS_CLOCK
        // Temperature data printing
        Serial.println("Temperature Data (T:" + String(tempTimestamp) + "):");
        Serial.print("Temperature: "); Serial.println(temperature);
        Serial.println();
#endif

#if HAS_PRESSURE
        // Pressure data printing
        Serial.println("Pressure Data (T:" + String(pressureTimestamp) + "):");
        Serial.print("Pressure 0: "); Serial.println(adc_value0);
        Serial.print("Pressure 1: "); Serial.println(adc_value1);
        Serial.println();
#endif
    }

    // JSON construction
    char jsonBuffer[1024];
    snprintf(jsonBuffer, sizeof(jsonBuffer),
        "{\"mpu_unit\": \"%d\", "
#if HAS_ENCODER
        "\"encoder\": {\"timestamp\": %llu, \"revolutions\": %.3f, \"rpm\": %.3f, \"speed\": %.3f, \"distance\": %.3f}, "
#endif
#if HAS_IMU
        "\"imu\": {\"timestamp\": %llu, \"acceleration\": {\"x\": %.5f, \"y\": %.5f, \"z\": %.5f}, \"gyroscope\": {\"x\": %.5f, \"y\": %.5f, \"z\": %.5f}}, "
#endif
#if HAS_CLOCK
        "\"temperature\": {\"timestamp\": %llu, \"value\": %.2f}, "
#endif
#if HAS_PRESSURE
        "\"pressure\": {\"timestamp\": %llu, \"adc_value0\": %d, \"adc_value1\": %d}, "
#endif
#if HAS_CLOCK
        "\"ptp\": {\"syncing\": \"%s\", \"status\": \"%s\", \"details\": \"Master: %s, Delay: %s\", \"difference\": \"%lld\", \"time_since_sync\": \"%lld\"}}",
#else
        "}",
#endif
        MPU_UNIT,
#if HAS_ENCODER
        encoderTimestamp, revolutions, rpm, speed, distance,
#endif
#if HAS_IMU
        imuTimestamp, ax_mss, ay_mss, az_mss, gx_rads, gy_rads, gz_rads,
#endif
#if HAS_CLOCK
        tempTimestamp, temperature,
#endif
#if HAS_PRESSURE
        pressureTimestamp, adc_value0, adc_value1,
#endif
#if HAS_CLOCK
        sync_clock ? "IN PROGRESS..." : "DONE/NOT STARTED", esp1588.GetLockStatus() ? "LOCKED" : "UNLOCKED", m.Healthy() ? "OK" : "BAD", esp1588.GetShortStatusString(),
        (int64_t)(getEpochMillis64(true) - 500000000), (int64_t)(getEpochMillis64() / 1000ULL - storedSyncTime)
#endif
    );

    // Send over UDP
    udp.beginPacket(udpAddress, udpPort);
    udp.print(jsonBuffer);
    udp.endPacket();
}

void PrintPTPInfo(ESP1588_Tracker& t) {
    const PTP_ANNOUNCE_MESSAGE& msg = t.GetAnnounceMessage();
    const PTP_PORTID& pid = t.GetPortIdentifier();

    SerialW.printf("    %s: ID ", t.IsMaster() ? "Master" : "Candidate");
    for (int i = 0; i < (int)(sizeof(pid.clockId) / sizeof(pid.clockId[0])); i++) {
        SerialW.printf("%02x ", pid.clockId[i]);
    }

    SerialW.printf(" Prio %3d ", msg.grandmasterPriority1);

    SerialW.printf(" %i-step", t.IsTwoStep() ? 2 : 1);

    SerialW.printf("\n");
}

void setup() {
    delay(250);
    SerialW.begin(115200);

#if HAS_ENCODER
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
#endif

    //delay(1000); // Wait a bit for serial connection to be established
    //SerialW.println("\nSCA50-400 Encoder Test");
    //SerialW.println("PPR: 400");
    //SerialW.println("Resolution: ±0.10mm\n");

    // Initialize Ethernet and register event handler
    WiFi.onEvent(WiFiEvent);
    ETH.begin();
    ETH.config(local_ip, gateway, subnet, dns);

    // Wait for Ethernet connection
    while (!ETH.linkUp()) {
        SerialW.println("Waiting for Ethernet connection...");
        delay(1000);
    }

    udp.begin(udpPort);           // Existing UDP for sending data (port 13233)
    udpCommand.begin(udpCommandPort);      // New UDP for receiving commands

#if HAS_IMU || HAS_CLOCK
    Wire.begin(I2C_SDA, I2C_SCL);
#endif

#if HAS_IMU
    // Initialize IMU (accelerometer and gyroscope)
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
#endif

#if HAS_CLOCK
    rtc.setClockMode(false); // Set 24h format
    rtc.enableOscillator(true, false, itr_freq); // enable the 1 Hz output on interrupt pin

    // Set up to handle interrupt from 1 Hz pin
    pinMode(RTC_ISR_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(RTC_ISR_PIN), rtcSecondInterrupt, FALLING);
#endif

    EEPROM.begin(512); // Initialize EEPROM with size 512 bytes
    EEPROM.get(0, storedSyncTime);   // Read the last stored sync time from EEPROM
    EEPROM.get(4, storedOffset);     // Read the offset from EEPROM
    Serial.printf("Stored sync time: %u\n", storedSyncTime);
    Serial.printf("Stored offset: %lld\n", storedOffset);

    esp1588.SetDomain(0);	// The domain of your PTP clock, 0 - 31
    esp1588.Begin();

#if HAS_PRESSURE
    // Initialize ADC
    Wire1.begin(I2C_1_SDA, I2C_1_SCL);
    SerialW.println("Initializing pressure sensor ADC...");
    if (!ads.begin(ADS1X15_ADDRESS, &Wire1)) {
        SerialW.println("Failed to initialize pressure sensor ADC, halting program.");
        while (1);
    }
#endif

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

    esp1588.Loop();	//this needs to be called OFTEN, at least several times per second but more is better. forget controlling program flow with delay() in your code.
    ESP1588_Tracker & m=esp1588.GetMaster();

    if (millis() - lastPrintTime >= printInterval) {
        dataPrint(m);
        lastPrintTime = millis();

        SerialW.printf("PTP status: %s   Master %s   Delay %s\n", esp1588.GetLockStatus()?"LOCKED":"UNLOCKED", m.Healthy()?"OK":"no", esp1588.GetShortStatusString());
        PrintPTPInfo(m);
        SerialW.println("\n");
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

    // Check for incoming UDP packets
    int packetSize = udpCommand.parsePacket();
    if (packetSize) {
        char packetBuffer[512];
        int len = udpCommand.read(packetBuffer, sizeof(packetBuffer) - 1);
        if (len > 0) {
            packetBuffer[len] = 0; // Null-terminate the string
        }

        // Parse the JSON data
        StaticJsonDocument<512> doc;
        DeserializationError error = deserializeJson(doc, packetBuffer);

        if (!error) {
            // Get the "units" array and "command" string
            JsonArray units = doc["units"];
            const char* command = doc["command"];

            // Check if MPU_UNIT is in the "units" array
            for (int unit : units) {
                if (unit == MPU_UNIT) {
                    if (strcmp(command, "reboot") == 0) {
                        ESP.restart();
                    }
#if HAS_ENCODER
                    else if (strcmp(command, "zero wait") == 0) {
                        noInterrupts();
                        resetCommandReceived = true;
                        interrupts();
                    }
                    else if (strcmp(command, "zero now") == 0) {
                        noInterrupts();

                        lastEncoderCount = lastEncoderCount - encoderCount; // Adjust the last count relative to zero
                        encoderCount = 0; // Reset the encoder count

                        resetOccurred = true; // Set the reset occurred flag
                        interrupts();
                    }
#endif
#if HAS_CLOCK
                    else if (strcmp(command, "sync") == 0) {
                        sync_clock = true;
                    }
#endif
#if HAS_IMU
                    else if (strcmp(command, "calibrate imu") == 0) {
                        // Add IMU calibration code here
                    }
#endif
                    break;
                }
            }
        }
    }
#if HAS_CLOCK
    uint64_t currentMillis = esp1588.GetEpochMillis64();
    if (sync_clock && esp1588.GetLockStatus() && m.Healthy() && esp1588.GetLastDiffMs() == 0 && currentMillis % 1000 == 0) {
        rtc.setEpoch(currentMillis / 1000, false);

        SerialW.printf("PTP status: %s   Master %s   Delay %s\n", esp1588.GetLockStatus() ? "LOCKED" : "UNLOCKED", m.Healthy() ? "OK" : "NOT OK", esp1588.GetShortStatusString());
        PrintPTPInfo(m);

        uint64_t offset = 0;

        SerialW.printf("\nRTC set. Saving unixtime as %lu and offset as %llu\n", currentMillis / 1000, offset);
        EEPROM.put(0, currentMillis / 1000); // Store time at address 0
        EEPROM.put(4, offset); // Store offset at address 4
        EEPROM.commit();
        storedSyncTime = currentMillis / 1000;
        storedOffset = offset;

        sync_clock = false;
    }
#endif
}
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
#include <math.h>

// The ESP32 unit number. Following 0:"X-axis Encoder", 1:"Z-axis Encoder - Main", 2:"Z-axis Encoder - Second", 3:"Surface Pressure Sensor" 
#define MPU_UNIT 2

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
#if MPU_UNIT == 1
#define DISTANCE_PER_COUNT 0.00002625352510  // meter per count. For linear sensors.
#define WHEEL_DIAMETER_MM 100 // Guesstimated
#define COUNTS_PER_REV (WHEEL_DIAMETER_MM * PI / 1000.0 / DISTANCE_PER_COUNT)

#elif MPU_UNIT == 2
#define DISTANCE_PER_COUNT 0.00002626184728  // meter per count. For linear sensors.
#define WHEEL_DIAMETER_MM 100 // Guesstimated
#define COUNTS_PER_REV (WHEEL_DIAMETER_MM * PI / 1000.0 / DISTANCE_PER_COUNT)

#else
#define PPR 400  // Pulses Per Revolution
#define WHEEL_DIAMETER_MM 95.52240427
#define COUNTS_PER_REV (PPR * 4)  // In quadrature mode
#define DISTANCE_PER_COUNT (WHEEL_DIAMETER_MM * PI / 1000.0 / COUNTS_PER_REV)
#endif

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
float accel[3] = {0, 0, 0};
float gyro[3] = {0, 0, 0};

const int iAx = 0, iAy = 1, iAz = 2, iGx = 3, iGy = 4, iGz = 5;

const int usDelay = 3150;   // empirical, to hold sampling to 200 Hz
const int NFast = 1000;     // the bigger, the better (but slower)
const int NSlow = 10000;
int LowValue[6], HighValue[6], Smoothed[6], LowOffset[6], HighOffset[6], Target[6];
int N;

float R_transform[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };
#endif

#if HAS_ENCODER
long currentCount;
float speed = 0, distance = 0;
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

unsigned long lastPrint = 0;

#if HAS_CLOCK
bool sync_clock = false;
uint32_t storedSyncTime = 0; // Store the last sync time read from EEPROM
int64_t storedOffset = 0;    // Variable to store the offset read from EEPROM
#endif

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
#if HAS_CLOCK
    // If there is a valid PTP lock, return that time
    uint64_t millis_comp = 0;

    ESP1588_Tracker& m = esp1588.GetMaster();
    if (esp1588.GetLockStatus() && m.Healthy() && abs(esp1588.GetLastDiffMs()) <= 100) {
        millis_comp = esp1588.GetEpochMillis64();
        if (!difference) {
            return millis_comp;
        }
    }

    DateTime now = RTClib::now();
    noInterrupts();
    shiftNs = esp_timer_get_time() * 1000ULL - localSecondOffsetNs;
    interrupts();
    // ShiftNs is the time since the last second, in nanoseconds, it shouldn't be negative or more than a second
    if (shiftNs < 0) {shiftNs = 0;} else if (shiftNs > 1000000000ULL) {shiftNs = 999999999ULL;}    

    if (difference) {
        return 5000000000ULL + (int64_t)(millis_comp - (((uint64_t)now.unixtime() * 1000ULL) + shiftNs / 1000000ULL)) + storedOffset / 1000000ULL;
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
    encoderCount -= lookup_table[index];

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

#if HAS_IMU
void computeRotationMatrix() {
    const int samples = 100;
    float sumx = 0, sumy = 0, sumz = 0;
    int16_t dummy_ax, dummy_ay, dummy_az, dummy_gx, dummy_gy, dummy_gz;
    for (int i = 0; i < samples; i++) {
         accelgyro.getMotion6(&dummy_ax, &dummy_ay, &dummy_az, &dummy_gx, &dummy_gy, &dummy_gz);
         float ax_temp = (float)dummy_ax / 16384.0 * g;
         float ay_temp = (float)dummy_ay / 16384.0 * g;
         float az_temp = (float)dummy_az / 16384.0 * g;
         sumx += ax_temp;
         sumy += ay_temp;
         sumz += az_temp;
         delay(10);
    }
    float ax_avg = sumx / samples;
    float ay_avg = sumy / samples;
    float az_avg = sumz / samples;
    
    // Normalize measured gravity vector:
    float norm = sqrt(ax_avg * ax_avg + ay_avg * ay_avg + az_avg * az_avg);
    if (norm < 1e-6) norm = 1; // avoid division by zero
    float gx0 = ax_avg / norm;
    float gy0 = ay_avg / norm;
    float gz0 = az_avg / norm;
    
    // Desired gravity direction is (0, 0, -1)
    float dx = 0.0;
    float dy = 0.0;
    float dz = -1.0;
    
    // Compute cross product between measured gravity and desired gravity
    float axis_x = gy0 * dz - gz0 * dy;  // = -gy0
    float axis_y = gz0 * dx - gx0 * dz;    // = gx0
    float axis_z = gx0 * dy - gy0 * dx;    // = 0
    float axis_norm = sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
    
    // If axis_norm is nearly zero, no rotation is needed
    if (axis_norm < 1e-6) {
       // Identity matrix
       R_transform[0][0] = 1; R_transform[0][1] = 0; R_transform[0][2] = 0;
       R_transform[1][0] = 0; R_transform[1][1] = 1; R_transform[1][2] = 0;
       R_transform[2][0] = 0; R_transform[2][1] = 0; R_transform[2][2] = 1;
       return;
    }
    axis_x /= axis_norm;
    axis_y /= axis_norm;
    axis_z /= axis_norm;
    
    // Compute angle between measured gravity and desired gravity
    float dot = gx0 * dx + gy0 * dy + gz0 * dz; // equals -gz0
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    float angle = acos(dot);
    
    // Rodrigues' rotation formula:
    float c = cos(angle);
    float s = sin(angle);
    float t = 1 - c;
    
    R_transform[0][0] = t * axis_x * axis_x + c;
    R_transform[0][1] = t * axis_x * axis_y - s * axis_z;
    R_transform[0][2] = t * axis_x * axis_z + s * axis_y;
    
    R_transform[1][0] = t * axis_x * axis_y + s * axis_z;
    R_transform[1][1] = t * axis_y * axis_y + c;
    R_transform[1][2] = t * axis_y * axis_z - s * axis_x;
    
    R_transform[2][0] = t * axis_x * axis_z - s * axis_y;
    R_transform[2][1] = t * axis_y * axis_z + s * axis_x;
    R_transform[2][2] = t * axis_z * axis_z + c;
    
    SerialW.println("Rotation matrix computed:");
    SerialW.printf("[%.3f, %.3f, %.3f]\n", R_transform[0][0], R_transform[0][1], R_transform[0][2]);
    SerialW.printf("[%.3f, %.3f, %.3f]\n", R_transform[1][0], R_transform[1][1], R_transform[1][2]);
    SerialW.printf("[%.3f, %.3f, %.3f]\n", R_transform[2][0], R_transform[2][1], R_transform[2][2]);

    // Save matrix to EEPROM
    EEPROM.put(12, 0xA5A5A5A5);
    EEPROM.put(16, R_transform);
    EEPROM.commit();
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
    distance = (float)currentCount * DISTANCE_PER_COUNT;

    // Calculate speed
    unsigned long timeDelta = encoderTimestamp - lastMeasurementTime;
    float countsPerSec = (float)(currentCount - lastEncoderCount) * (1000.0 / timeDelta);
    speed = countsPerSec * DISTANCE_PER_COUNT;

    if (!isfinite(speed)) {
        speed = 0.0;
    }

    lastEncoderCount = currentCount;
    lastMeasurementTime = encoderTimestamp;
#endif

#if HAS_IMU
    // IMU data capture
    imuTimestamp = getEpochMillis64();
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert raw acceleration data to m/s², and raw gyroscope data to radians per second
    accel[0] = (float)ax / 16384.0 * g;
    accel[1] = (float)ay / 16384.0 * g;
    accel[2] = (float)az / 16384.0 * g;
    gyro[0] = (float)gx / 131.0 * (PI / 180.0);
    gyro[1] = (float)gy / 131.0 * (PI / 180.0);
    gyro[2] = (float)gz / 131.0 * (PI / 180.0);

    float accel_rot[3] = {0, 0, 0};
    float gyro_rot[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        accel_rot[i] = R_transform[i][0]*accel[0] + R_transform[i][1]*accel[1] + R_transform[i][2]*accel[2];
        gyro_rot[i] = R_transform[i][0]*gyro[0] + R_transform[i][1]*gyro[1] + R_transform[i][2]*gyro[2];
    }
    // In the rotated frame gravity should be (0,0,-g).
    // Subtract gravity from the acceleration vector so that stationary the net acceleration is nearly zero.
    // accel_rot[2] += g;  // subtracting (0,0,-g) is equivalent to adding g to the z-axis.
    
    accel[0] = accel_rot[0]; accel[1] = accel_rot[1]; accel[2] = accel_rot[2];
    gyro[0] = gyro_rot[0];   gyro[1] = gyro_rot[1];   gyro[2] = gyro_rot[2];
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
        Serial.print("Counts: "); Serial.print(currentCount); Serial.print("\t");
        Serial.print("Speed (m/s): "); Serial.print(speed, 5); Serial.print("\t");
        Serial.print("Distance (m): "); Serial.print(distance, 5); Serial.println();
        Serial.println();
#endif

#if HAS_IMU
        // IMU data printing
        Serial.println("IMU Data (T:" + String(imuTimestamp) + "):");
        Serial.print("Acceleration (m/s²): ");
        Serial.print("X="); Serial.print(accel[0], 5); Serial.print("\t");
        Serial.print("Y="); Serial.print(accel[1], 5); Serial.print("\t");
        Serial.print("Z="); Serial.print(accel[2], 5); Serial.println();
        Serial.print("Gyroscope (rad/s): ");
        Serial.print("X="); Serial.print(gyro[0], 5); Serial.print("\t");
        Serial.print("Y="); Serial.print(gyro[1], 5); Serial.print("\t");
        Serial.print("Z="); Serial.print(gyro[2], 5); Serial.println();
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
        "{\"mpu_unit\": \"%d\", \"packet_number\": %d, "
#if HAS_ENCODER
        "\"encoder\": {\"timestamp\": %llu, \"counts\": %ld, \"speed\": %.5f, \"distance\": %.5f}, "
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
        MPU_UNIT, lastPrint++,
#if HAS_ENCODER
        encoderTimestamp, currentCount, speed, distance,
#endif
#if HAS_IMU
        imuTimestamp, accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2],
#endif
#if HAS_CLOCK
        tempTimestamp, temperature,
#endif
#if HAS_PRESSURE
        pressureTimestamp, adc_value0, adc_value1,
#endif
#if HAS_CLOCK
        sync_clock ? "IN PROGRESS..." : "DONE/NOT STARTED", esp1588.GetLockStatus() ? "LOCKED" : "UNLOCKED", m.Healthy() ? "OK" : "BAD", esp1588.GetShortStatusString(),
        (int64_t)(getEpochMillis64(true) - 5000000000ULL), (int64_t)(getEpochMillis64() / 1000ULL - storedSyncTime)
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

// Reset IMU Functions
#if HAS_IMU
void GetSmoothed() { 
    int16_t RawValue[6];
    int i;
    long Sums[6];
    for (i = iAx; i <= iGz; i++) { Sums[i] = 0; }

    for (i = 1; i <= N; i++) { // get sums
        accelgyro.getMotion6(&RawValue[iAx], &RawValue[iAy], &RawValue[iAz], 
                             &RawValue[iGx], &RawValue[iGy], &RawValue[iGz]);
        delayMicroseconds(usDelay);
        for (int j = iAx; j <= iGz; j++)
          Sums[j] = Sums[j] + RawValue[j];
    } // get sums

    for (i = iAx; i <= iGz; i++) { Smoothed[i] = (Sums[i] + N/2) / N ; }
} // GetSmoothed
void SetOffsets(int TheOffsets[6]) {
    accelgyro.setXAccelOffset(TheOffsets [iAx]);
    accelgyro.setYAccelOffset(TheOffsets [iAy]);
    accelgyro.setZAccelOffset(TheOffsets [iAz]);
    accelgyro.setXGyroOffset (TheOffsets [iGx]);
    accelgyro.setYGyroOffset (TheOffsets [iGy]);
    accelgyro.setZGyroOffset (TheOffsets [iGz]);
} // SetOffsets
void PullBracketsIn() {
    boolean AllBracketsNarrow;
    boolean StillWorking;
    int NewOffset[6];
  
    AllBracketsNarrow = false;
    StillWorking = true;
    while (StillWorking) {
        StillWorking = false;
        if (AllBracketsNarrow && (N == NFast)) {
            N = NSlow;
        } else { AllBracketsNarrow = true; }// tentative
        
        for (int i = iAx; i <= iGz; i++) { 
            if (HighOffset[i] <= (LowOffset[i]+1)) {
                NewOffset[i] = LowOffset[i];
            } else { // binary search
                StillWorking = true;
                NewOffset[i] = (LowOffset[i] + HighOffset[i]) / 2;
                if (HighOffset[i] > (LowOffset[i] + 10)) { AllBracketsNarrow = false; }
            } // binary search
        }

        SetOffsets(NewOffset);
        GetSmoothed();
        for (int i = iAx; i <= iGz; i++) { // closing in
            if (Smoothed[i] > Target[i]) { // use lower half
                HighOffset[i] = NewOffset[i];
                HighValue[i] = Smoothed[i];
            } else { // use upper half
                LowOffset[i] = NewOffset[i];
                LowValue[i] = Smoothed[i];
            } // use upper half
        } // closing in
    } // still working
} // PullBracketsIn
void PullBracketsOut() {
    boolean Done = false;
    int NextLowOffset[6];
    int NextHighOffset[6];

    while (!Done) {
        Done = true;
        SetOffsets(LowOffset);
        GetSmoothed();
        for (int i = iAx; i <= iGz; i++) { // got low values
            LowValue[i] = Smoothed[i];
            if (LowValue[i] >= Target[i]) { 
                Done = false;
                NextLowOffset[i] = LowOffset[i] - 1000;
            } else { NextLowOffset[i] = LowOffset[i]; }
        } // got low values
      
        SetOffsets(HighOffset);
        GetSmoothed();
        for (int i = iAx; i <= iGz; i++) { // got high values
            HighValue[i] = Smoothed[i];
            if (HighValue[i] <= Target[i]) {
                Done = false;
                NextHighOffset[i] = HighOffset[i] + 1000;
            } else { NextHighOffset[i] = HighOffset[i]; }
        } // got high values

        for (int i = iAx; i <= iGz; i++) {
            LowOffset[i] = NextLowOffset[i];   // had to wait until ShowProgress done
            HighOffset[i] = NextHighOffset[i]; // ..
        }
    } // keep going
} // PullBracketsOut
#endif


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
    
    delay(1000); // Wait a bit for serial connection to be established
    if (MPU_UNIT == 0) {
        SerialW.println("\nSCA50-400 Encoder Test");
        SerialW.println("PPR: 400");
        SerialW.println("Resolution: ±0.10mm\n");
    } else {
        SerialW.println("\nWDS-7500-P115 Wire Sensor Test");
        SerialW.println("Resolution: ±0.03mm & ±0.02% FSO\n");
    }
#endif

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

#if HAS_IMU || HAS_CLOCK
    EEPROM.begin(512); // Initialize EEPROM with size 512 bytes
#endif

#if HAS_CLOCK
    EEPROM.get(0, storedSyncTime);   // Read the last stored sync time from EEPROM
    EEPROM.get(4, storedOffset);     // Read the offset from EEPROM
    SerialW.printf("Stored sync time: %u\n", storedSyncTime);
    SerialW.printf("Stored offset: %lld\n", storedOffset);

    esp1588.SetDomain(0);	// The domain of your PTP clock, 0 - 31
    esp1588.Begin();
#endif

#if HAS_IMU
    // Load rotation matrix if valid
    uint32_t flag;
    EEPROM.get(12, flag);
    if (flag == 0xA5A5A5A5) {
        EEPROM.get(16, R_transform);
    }
#endif

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

#if HAS_CLOCK
    esp1588.Loop();	//this needs to be called OFTEN, at least several times per second but more is better. forget controlling program flow with delay() in your code.
    ESP1588_Tracker & m = esp1588.GetMaster();
#endif

    if (millis() - lastPrintTime >= printInterval) {
        dataPrint(m);
        lastPrintTime = millis();

#if HAS_CLOCK
        SerialW.printf("PTP status: %s   Master %s   Delay %s\n", esp1588.GetLockStatus()?"LOCKED":"UNLOCKED", m.Healthy()?"OK":"no", esp1588.GetShortStatusString());
        PrintPTPInfo(m);
#endif
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
                        Serial.println("Calibrating IMU (this may take several minutes)...");

                        // Informing the user about the calibration process
                        char jsonBuffer[1024];
                        snprintf(jsonBuffer, sizeof(jsonBuffer),
                            "{\"mpu_unit\": \"%d\", "
                            "\"imu\": {\"info\": \"%s\", \"info_1\": \"%s\", \"info_2\": \"%s\", \"info_3\": \"%s\", \"info_4\": \"%s\"}}",
                            MPU_UNIT,
                            "Calibrating IMU...", "Please do not", "move the unit.", "(This takes", "several minutes)"
                            );
                        udp.beginPacket(udpAddress, udpPort);
                        udp.print(jsonBuffer);
                        udp.endPacket();

                        accelgyro.CalibrateAccel(6);
                        accelgyro.CalibrateGyro(6);
                        accelgyro.CalibrateAccel(1);
                        accelgyro.CalibrateGyro(1);
                        accelgyro.CalibrateAccel(1);
                        accelgyro.CalibrateGyro(1);
                        accelgyro.CalibrateAccel(1);
                        accelgyro.CalibrateGyro(1);
                        accelgyro.CalibrateAccel(1);
                        accelgyro.CalibrateGyro(1);
                        for (int i = iAx; i <= iGz; i++) {
                            Target[i] = 0; // must fix for ZAccel 
                            HighOffset[i] = 0;
                            LowOffset[i] = 0;
                        } // set targets and initial guesses
                        Target[iAy] = 16384;
                        N = NFast;
                        
                        PullBracketsOut();
                        PullBracketsIn();

                        SerialW.println("IMU Calibration complete.");
                        
                        // Compute the transformation rotation matrix using current stationary readings.
                        computeRotationMatrix();
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
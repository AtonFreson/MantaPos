#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// Pin definitions for SCA50 differential signals
#define ENCODER_PIN_A_POS 14  // A
#define ENCODER_PIN_A_NEG 15  // A-
#define ENCODER_PIN_B_POS 32  // B
#define ENCODER_PIN_B_NEG 33  // B-
#define INDEX_PIN_Z_POS 35    // Z
#define INDEX_PIN_Z_NEG 36    // Z-
 
// Configuration
#define PPR 400  // Pulses Per Revolution
#define COUNTS_PER_REV (PPR * 4)  // In quadrature mode
#define WHEEL_DIAMETER_MM 82.53
#define DISTANCE_PER_REV (WHEEL_DIAMETER_MM * PI / 1000.0)  // meters per revolution
#define DISTANCE_PER_COUNT (DISTANCE_PER_REV / COUNTS_PER_REV)

volatile long encoderCount = 0;
long lastEncoderCount = 0;
unsigned long lastPrintTime = 0;
unsigned long lastMeasurementTime = 0;
const unsigned long printInterval = 10; // 10ms update rate
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
    
    bool isConnected() {
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
    
    // Attach interrupts - using CHANGE for both edges of the signals
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

    if (currentTime - lastPrintTime >= printInterval) {
        noInterrupts();  // Disable interrupts while reading volatile
        long currentCount = encoderCount;
        interrupts();

        unsigned long timeDelta = currentTime - lastMeasurementTime;

        if (currentCount != lastEncoderCount && timeDelta > 0) {
            float revolutions = (float)currentCount / COUNTS_PER_REV;
            float distance = currentCount * DISTANCE_PER_COUNT;

            // Calculate speed
            float countsPerSec = (float)(currentCount - lastEncoderCount) * (1000.0 / timeDelta);
            float rpm = (countsPerSec * 60.0) / COUNTS_PER_REV;
            float speed_m_per_s = countsPerSec * DISTANCE_PER_COUNT;

            char buffer[150];
            snprintf(buffer, sizeof(buffer), 
                    "Rev: %8.3f  RPM: %8.3f  Speed (m/s): %8.3f  Dist (m): %9.4f", 
                    revolutions, rpm, speed_m_per_s, distance);
            SerialW.println(buffer);

            // Prepare UDP packet
            udp.beginPacket(udpAddress, udpPort);
            udp.print(buffer);
            udp.endPacket();

            lastEncoderCount = currentCount;
            lastMeasurementTime = currentTime;
        }
        lastPrintTime = currentTime;
    }
}
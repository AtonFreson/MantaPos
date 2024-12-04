#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <DS3231.h>
#include <EEPROM.h> // Include EEPROM library

#define RTC_ISR_PIN 2

// Create DS3231 object
DS3231 rtc;

// Network configuration
IPAddress local_ip(169, 254, 178, 100);  
IPAddress gateway(0, 0, 0, 0);           
IPAddress subnet(255, 255, 0, 0);        
IPAddress dns(0, 0, 0, 0);               

const unsigned int localEventPort = 319;   
const unsigned int localGeneralPort = 320; 
WiFiUDP UdpEvent;   
WiFiUDP UdpGeneral; 

IPAddress ptpMulticastIP(224, 0, 1, 129);
IPAddress masterIP(169, 254, 178, 87);

// PTP message types
#define SYNC           0x00
#define FOLLOW_UP      0x08

#define PTP_MSG_SIZE 128
byte ptpMsgBuffer[PTP_MSG_SIZE];

// Timing variables
uint64_t t1 = 0;
uint64_t t2 = 0;
int64_t drift = 0;

bool ethernetConnected = false;
bool udpInitialized = false;

uint32_t storedSyncTime = 0; // Store the last sync time read from EEPROM
int64_t storedOffset = 0;    // Variable to store the offset read from EEPROM

unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL = 5000; // 5 seconds in milliseconds

bool readyToPrint = false;
int64_t lastCalculatedDrift = 0;

void WiFiEvent(arduino_event_t *event) {
    switch (event->event_id) {
        case ARDUINO_EVENT_ETH_START:
            Serial.println("ETH: Started");
            ETH.setHostname("esp32-drift-checker");
            break;
        case ARDUINO_EVENT_ETH_CONNECTED:
            Serial.println("ETH: Connected");
            ethernetConnected = true;
            break;
        case ARDUINO_EVENT_ETH_GOT_IP:
            Serial.printf("ETH: IP: %s\n", ETH.localIP().toString().c_str());
            Serial.printf("ETH: Speed: %dMbps\n", ETH.linkSpeed());
            Serial.printf("ETH: Mode: %s\n", ETH.fullDuplex() ? "Full Duplex" : "Half Duplex");
            initializeUDP();
            break;
        case ARDUINO_EVENT_ETH_DISCONNECTED:
            Serial.println("ETH: Disconnected");
            ethernetConnected = false;
            udpInitialized = false;
            break;
        case ARDUINO_EVENT_ETH_STOP:
            Serial.println("ETH: Stopped");
            ethernetConnected = false;
            udpInitialized = false;
            break;
        default:
            break;
    }
}

void initializeUDP() {
    bool success = true;

    if (!UdpEvent.begin(localEventPort)) {
        Serial.println("Event UDP initialization failed");
        success = false;
    }
    
    if (!UdpGeneral.begin(localGeneralPort)) {
        Serial.println("General UDP initialization failed");
        success = false;
    }

    if (!UdpEvent.beginMulticast(ptpMulticastIP, localEventPort)) {
        Serial.println("Event multicast join failed");
        success = false;
    }
    
    if (!UdpGeneral.beginMulticast(ptpMulticastIP, localGeneralPort)) {
        Serial.println("General multicast join failed");
        success = false;
    }

    if (success) {
        udpInitialized = true;
        Serial.println("UDP initialized");
    } else {
        udpInitialized = false;
        Serial.println("UDP initialization failed");
    }
}

// Use esp_timer_get_time() to get the current time from ESP32 in microseconds since boot
uint64_t getCurrentTimeInNanos() {
    // Get current time from the RTC
    DateTime now = RTClib::now();
    noInterrupts();
    shiftNs = esp_timer_get_time() * 1000ULL - localSecondOffsetNs;
    interrupts();

    return (((uint64_t)now.unixtime() * 1000000000ULL) + shiftNs); // Convert to nanoseconds (x9)
}

uint64_t extractTimestamp(byte *buffer, int startIndex) {
    uint64_t seconds = 0;
    for (int i = 0; i < 6; i++) {
        seconds = (seconds << 8) | buffer[startIndex + i];
    }
    uint32_t nanoseconds = 0;
    for (int i = 0; i < 4; i++) {
        nanoseconds = (nanoseconds << 8) | buffer[startIndex + 6 + i];
    }
    return (seconds * 1000000000ULL) + nanoseconds;
}

void printDriftInfo() {
    // Get current time in seconds
    uint32_t currentTime = t2 / 1000000000ULL;
    uint32_t timeSinceSync = currentTime - storedSyncTime;

    // Calculate time components
    uint32_t days = timeSinceSync / 86400;
    uint32_t hours = (timeSinceSync % 86400) / 3600;
    uint32_t minutes = (timeSinceSync % 3600) / 60;
    uint32_t seconds = timeSinceSync % 60;

    // Calculate drift in milliseconds with rounding
    int64_t drift_ms;
    if (lastCalculatedDrift >= 0) {
        drift_ms = (lastCalculatedDrift + 500000) / 1000000LL;
    } else {
        drift_ms = (lastCalculatedDrift - 500000) / 1000000LL;
    }

    // Calculate drift in ppm
    double drift_ppm = 0.0;
    if (timeSinceSync > 0) {
        drift_ppm = ((double)lastCalculatedDrift / (double)(timeSinceSync * 1000000000ULL)) * 1e6;
    }

    // Print time since sync
    Serial.printf("Time since last sync: %u days, %u hours, %u minutes, %u seconds\n", 
                 days, hours, minutes, seconds);

    // Print drift with ppm
    if (drift_ms > 0) {
        Serial.printf("ESP32 is slow by %lld ms (%.6f ppm)\n", drift_ms, drift_ppm);
    } else if (drift_ms < 0) {
        Serial.printf("ESP32 is fast by %lld ms (%.6f ppm)\n", -drift_ms, -drift_ppm);
    } else {
        Serial.println("ESP32 clock is synchronized to <1 ms with master");
    }
}

void handleSyncMessage(byte *buffer) {
    uint16_t recvSequence = (buffer[30] << 8) | buffer[31];
    if (recvSequence <= lastRecvSequence) return;
    t2 = getCurrentTimeInNanos();

    lastRecvSequence = recvSequence;
}

void handleFollowUpMessage(byte *buffer, int64_t drift_start_offset) {
    t1 = extractTimestamp(buffer, 34);
    if (t2 != 0) {
        lastCalculatedDrift = (int64_t)(t1 - t2) + drift_start_offset;
        readyToPrint = true;
        t1 = t2 = 0;
    }
}

void processPTPEventMessage(byte *buffer, int size) {
    byte messageType = buffer[0] & 0x0F;
    if (messageType == SYNC) handleSyncMessage(buffer);
}

void processPTPGeneralMessage(byte *buffer, int size, int64_t drift_start_offset) {
    byte messageType = buffer[0] & 0x0F;
    if (messageType == FOLLOW_UP) {
        handleFollowUpMessage(buffer, drift_start_offset);
    }
}

void rtcSecondInterrupt () {
    localSecondOffsetNs = esp_timer_get_time() * 1000ULL;
}


void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("DS3231 Drift Checker Starting");

    Wire.begin();
    rtc.setClockMode(false); // Set 24h format
    rtc.enableOscillator(true, false, itr_freq); // enable the 1 Hz output on interrupt pin

    // set up to handle interrupt from 1 Hz pin
    pinMode (RTC_ISR_PIN, INPUT_PULLUP);
    attachInterrupt (digitalPinToInterrupt (RTC_ISR_PIN), rtcSecondInterrupt, RISING);

    // Initialize EEPROM and read stored sync time and offset
    EEPROM.begin(512); // Initialize EEPROM with size 512 bytes
    EEPROM.get(0, storedSyncTime);   // Read the last stored sync time from EEPROM
    EEPROM.get(4, storedOffset);     // Read the offset from EEPROM

    WiFi.onEvent(WiFiEvent);
    ETH.begin();
    ETH.config(local_ip, gateway, subnet, dns);

    Serial.printf("Stored sync time: %u\n", storedSyncTime);
    Serial.printf("Stored offset: %lld\n", storedOffset);
}

void loop() {
    if (!ethernetConnected || !udpInitialized) {
        if (!ETH.linkUp()) ESP.restart();
        initializeUDP();
    }

    int packetSize = UdpEvent.parsePacket();
    if (packetSize) {
        UdpEvent.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPEventMessage(ptpMsgBuffer, packetSize);
    }

    int packetSize = UdpGeneral.parsePacket();
    if (packetSize) {
        UdpGeneral.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPGeneralMessage(ptpMsgBuffer, packetSize, storedOffset);
    }

    // Handle printing with delay
    unsigned long currentMillis = millis();
    if (currentMillis - lastPrintTime >= PRINT_INTERVAL) {
        lastPrintTime = currentMillis;
        if (readyToPrint) {
            printDriftInfo();
            readyToPrint = false;
        } else {
            Serial.println("Waiting for data packets...");
        }
    }
}

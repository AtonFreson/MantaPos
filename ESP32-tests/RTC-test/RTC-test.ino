#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <DS3231.h>
#include <EEPROM.h> // Include EEPROM library

DS3231 rtc;
bool rtcInitialized = false;

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
#define DELAY_REQ      0x01
#define FOLLOW_UP      0x08
#define DELAY_RESP     0x09
#define ANNOUNCE       0x0B

#define PTP_MSG_SIZE 128
byte ptpMsgBuffer[PTP_MSG_SIZE];

// Timing variables
uint64_t t1 = 0;
uint64_t t2 = 0;
uint64_t t3 = 0;
uint64_t t4 = 0;
int64_t offset = 0;
uint64_t roundTripDelay = 0;

uint16_t sequenceId = 0;
uint16_t lastRecvSequence = 0;
byte clockId[8] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};

bool ethernetConnected = false;
bool udpInitialized = false;
unsigned long lastPacketSent = 0;
unsigned long lastPacketReceived = 0;
const unsigned long NETWORK_TIMEOUT = 30000; // 30 seconds

uint32_t lastSyncTime = 0;          // Store the last sync time in unix format
uint32_t storedSyncTime = 0;        // Store the last sync time read from EEPROM

unsigned long rtcUpdateStartTime = 0; // Track when RTC updates started
bool rtcUpdateExpired = false;        // Flag to indicate if RTC update period has expired

void WiFiEvent(arduino_event_t *event) {
    switch (event->event_id) {
        case ARDUINO_EVENT_ETH_START:
            Serial.println("ETH: Started");
            ETH.setHostname("esp32-ptp");
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
        Serial.println("UDP fully initialized");
        sendTestPacket();
    } else {
        udpInitialized = false;
        Serial.println("UDP initialization failed");
    }
}

void sendTestPacket() {
    if (!udpInitialized) return;
    
    if (UdpEvent.beginPacket(masterIP, localEventPort)) {
        const uint8_t testMsg[] = {'T', 'E', 'S', 'T'};
        UdpEvent.write(testMsg, 4);
        if (UdpEvent.endPacket()) {
            Serial.println("Test packet sent successfully");
            lastPacketSent = millis();
        } else {
            Serial.println("Failed to send test packet");
        }
    }
}

void buildPTPHeader(byte messageType) {
    memset(ptpMsgBuffer, 0, PTP_MSG_SIZE);
    
    ptpMsgBuffer[0] = (0x02 << 4) | messageType;
    ptpMsgBuffer[1] = 0x02;
    ptpMsgBuffer[30] = (sequenceId >> 8) & 0xFF;
    ptpMsgBuffer[31] = sequenceId & 0xFF;
    memcpy(&ptpMsgBuffer[20], clockId, 8);
}

void sendDelayReq() {
    unsigned long tempSendPacket = millis();

    if (!ethernetConnected || !udpInitialized) {
        Serial.println("Network not ready for DELAY_REQ");
        return;
    }

    buildPTPHeader(DELAY_REQ);
    t3 = getCurrentTimeInNanos();
    
    uint64_t seconds = t3 / 1000000000ULL;
    uint32_t nanoseconds = t3 % 1000000000ULL;
    
    for (int i = 0; i < 6; i++) {
        ptpMsgBuffer[34 + i] = (seconds >> ((5 - i) * 8)) & 0xFF;
    }
    for (int i = 0; i < 4; i++) {
        ptpMsgBuffer[40 + i] = (nanoseconds >> ((3 - i) * 8)) & 0xFF;
    }

    if (!UdpEvent.beginPacket(masterIP, localEventPort)) {
        Serial.println("Begin packet failed");
        return;
    }
    UdpEvent.write(ptpMsgBuffer, 44);
    if (UdpEvent.endPacket()) {
        lastPacketSent = tempSendPacket;
        sequenceId++;
    } else {
        Serial.println("End packet failed");
    }
}

void handleSyncMessage(byte *buffer) {
    uint16_t recvSequence = (buffer[30] << 8) | buffer[31];
    if (recvSequence <= lastRecvSequence) return;
    
    lastRecvSequence = recvSequence;
    t2 = getCurrentTimeInNanos();
}

void handleFollowUpMessage(byte *buffer) {
    t1 = extractTimestamp(buffer, 34);

    if (!rtcUpdateExpired) {
        // Check if RTC is not initialized or time difference is significant
        if (!rtcInitialized || abs((int64_t)(t1 - getCurrentTimeInNanos())) > 1000000000ULL) {
            rtc.setEpoch(t1 / 1000000000ULL, false); // Set RTC time in seconds
            rtcInitialized = true;
        }
    }
    
    calculateOffsetAndDelay();
}

void handleDelayRespMessage(byte *buffer) {
    t4 = extractTimestamp(buffer, 34);
    calculateOffsetAndDelay();
}

void processPTPEventMessage(byte *buffer, int size) {
    byte messageType = buffer[0] & 0x0F;
    if (messageType == SYNC) handleSyncMessage(buffer);
}

void processPTPGeneralMessage(byte *buffer, int size) {
    byte messageType = buffer[0] & 0x0F;
    if (messageType == FOLLOW_UP) handleFollowUpMessage(buffer);
    else if (messageType == DELAY_RESP) handleDelayRespMessage(buffer);
}

void calculateOffsetAndDelay() {
    if (t1 && t2 && t3 && t4) {
        int64_t master_to_slave = (int64_t)(t2 - t1);
        int64_t slave_to_master = (int64_t)(t3 - t4);

        offset = (master_to_slave + slave_to_master) / 2;
        roundTripDelay = (master_to_slave - slave_to_master) / 2;

        Serial.printf("\nt1: %llu, t2: %llu, t3: %llu, t4: %llu\n", t1, t2, t3, t4);
        Serial.printf("Master to Slave: %lld, Slave to Master: %lld\n", master_to_slave, slave_to_master);
        Serial.printf("Offset: %lld ns (%f seconds)\n", offset, (double)offset / 1000000000.0);
        Serial.printf("Delay: %lld ns\n", roundTripDelay);

        adjustLocalClock(offset);
        t1 = t2 = t3 = t4 = 0;
    }
}

void adjustLocalClock(int64_t offset) {
    if (rtcUpdateExpired || !rtcInitialized) return;

    // Start RTC update timer
    if (rtcUpdateStartTime == 0) {
        rtcUpdateStartTime = millis();
    }

    // Only update RTC for 30 seconds
    if (millis() - rtcUpdateStartTime < 30000UL) {
        uint64_t correctedTime = t2 - offset;
        lastSyncTime = correctedTime / 1000000000ULL;
        rtc.setEpoch(lastSyncTime, false);
        
        // Save time and offset to EEPROM
        storedSyncTime = lastSyncTime;
        EEPROM.put(0, storedSyncTime); // Store time at address 0
        EEPROM.put(4, offset);         // Store offset at address 4
        EEPROM.commit();
        Serial.println("Sync Time and Offset stored to EEPROM");
    } else {
        // Stop updating RTC after 2 minutes
        rtcUpdateExpired = true;
    }
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

// Use unixtime() to get the current time from RTC in seconds since 1970
uint64_t getCurrentTimeInNanos() {
    DateTime now = RTClib::now();
    return (uint64_t)now.unixtime() * 1000000000ULL; // Convert to nanoseconds (x9)
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("PTP Client Starting");
    WiFi.onEvent(WiFiEvent);
    ETH.begin();
    ETH.config(local_ip, gateway, subnet, dns);

    Wire.begin();
    rtc.setClockMode(false); // Set 24h format
    
    rtc.setEpoch(1733110618); // Set rtc to current time, Fri Nov 29 2024 20:03:38 UTC

    EEPROM.begin(512); // Initialize EEPROM with size 512 bytes
    EEPROM.get(0, storedSyncTime); // Read the last stored sync time from EEPROM
    EEPROM.get(4, offset); // Read the stored offset from EEPROM
}

void loop() {
    if (rtcUpdateExpired) {
        Serial.println("RTC update period has expired, waiting for reset...");
        Serial.printf("Final stored sync time: %u\n", storedSyncTime);
        Serial.printf("Final stored offset: %lld\n", offset);
        while (true) {
            delay(1000);
        }
    }

    if (!ethernetConnected || !udpInitialized) {
        if (!ETH.linkUp()) ESP.restart();
        initializeUDP();
    }

    int packetSize = UdpEvent.parsePacket();
    if (packetSize) {
        UdpEvent.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPEventMessage(ptpMsgBuffer, packetSize);
    }
    
    packetSize = UdpGeneral.parsePacket();
    if (packetSize) {
        UdpGeneral.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPGeneralMessage(ptpMsgBuffer, packetSize);
    }

    if (millis() - lastPacketSent > 1000) sendDelayReq();
}

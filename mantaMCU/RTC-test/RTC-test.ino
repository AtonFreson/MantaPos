#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <DS3231.h>

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
        lastPacketSent = millis();
        sequenceId++;
    } else {
        Serial.println("End packet failed");
    }
}

void handleSyncMessage(byte *buffer) {
    uint16_t recvSequence = (buffer[30] << 8) | buffer[31];
    if (recvSequence <= lastRecvSequence) return;
    
    lastRecvSequence = recvSequence;
    t1 = extractTimestamp(buffer, 34);
    t2 = getCurrentTimeInNanos();
}

void handleFollowUpMessage(byte *buffer) {
    t1 = extractTimestamp(buffer, 34);
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
        int64_t master_to_slave = t2 - t1;
        int64_t slave_to_master = t3 - t4;

        offset = (master_to_slave + slave_to_master) / 2;
        roundTripDelay = (master_to_slave - slave_to_master) / 2;

        Serial.printf("\nt1: %llu, t2: %llu, t3: %llu, t4: %llu\n", t1, t2, t3, t4); 
        Serial.printf("Offset: %lld ns\n", offset);
        Serial.printf("Delay: %llu ns\n", roundTripDelay);
        adjustLocalClock(offset);
        t1 = t2 = t3 = t4 = 0;
    }
}

void adjustLocalClock(int64_t offset) {
    if (!rtcInitialized) return;
    RTCDateTime now = rtc.getDateTime();
    uint64_t currentTime = now.unixtime * 1000000000ULL;
    uint64_t adjustedTime = currentTime + offset;
    rtc.setDateTime(adjustedTime / 1000000000ULL);
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

uint64_t getCurrentTimeInNanos() {
    if (!rtcInitialized) return 0;
    RTCDateTime now = rtc.getDateTime();
    return (uint64_t)now.unixtime * 1000000000ULL;
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("PTP Client Starting");
    WiFi.onEvent(WiFiEvent);
    ETH.begin();
    ETH.config(local_ip, gateway, subnet, dns);

    Wire.begin();
    rtc.begin();
    rtcInitialized = rtc.isReady();
    Serial.printf("RTC initialized: %d\n", rtcInitialized);
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
    
    packetSize = UdpGeneral.parsePacket();
    if (packetSize) {
        UdpGeneral.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPGeneralMessage(ptpMsgBuffer, packetSize);
    }

    if (millis() - lastPacketSent > 1000) sendDelayReq();
}

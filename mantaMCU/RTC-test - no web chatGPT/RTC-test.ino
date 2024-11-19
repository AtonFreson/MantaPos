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
#define PDELAY_REQ     0x02
#define PDELAY_RESP    0x03

#define PTP_MSG_SIZE 128
#define PTP_HEADER_SIZE 34
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

#define USE_OFFSET_CHECK 0 // Set to 1 to use offset sanity check
#define MAX_OFFSET 1000000000LL // 1 second max offset
#define MIN_OFFSET -1000000000LL // -1 second min offset

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
        // Send test packet
        sendTestPacket();
    } else {
        udpInitialized = false;
        Serial.println("UDP initialization failed");
    }
}

void sendTestPacket() {
    if (!udpInitialized) return;
    
    if (UdpEvent.beginPacket(masterIP, localEventPort)) {
        // Convert string to uint8_t array
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
  ptpMsgBuffer[2] = 0x00;
  ptpMsgBuffer[3] = 0x2C;
  ptpMsgBuffer[4] = 0x00;
  ptpMsgBuffer[5] = 0x00;
  
  ptpMsgBuffer[6] = 0x00;
  ptpMsgBuffer[7] = 0x00;
  
  for(int i = 8; i < 16; i++) {
    ptpMsgBuffer[i] = 0x00;
  }
  
  memcpy(&ptpMsgBuffer[20], clockId, 8);
  ptpMsgBuffer[28] = 0x00;
  ptpMsgBuffer[29] = 0x01;
  
  ptpMsgBuffer[30] = (sequenceId >> 8) & 0xFF;
  ptpMsgBuffer[31] = sequenceId & 0xFF;
  
  ptpMsgBuffer[32] = (messageType == DELAY_REQ) ? 0x01 : 0x05;
  ptpMsgBuffer[33] = 0x7F;
}

void sendDelayReq() {
    if (!ethernetConnected || !udpInitialized) {
        Serial.println("Network not ready for DELAY_REQ");
        return;
    }

    if(ETH.localIP().toString() == "0.0.0.0") {
        Serial.println("No IP yet, skipping DELAY_REQ");
        return;
    }

    buildPTPHeader(DELAY_REQ);
    
    t3 = getCurrentTimeInNanos();
    uint64_t seconds = t3 / 1000000000ULL;
    uint32_t nanoseconds = t3 % 1000000000ULL;
    
    for(int i = 0; i < 6; i++) {
        ptpMsgBuffer[34 + i] = (seconds >> ((5-i)*8)) & 0xFF;
    }
    for(int i = 0; i < 4; i++) {
        ptpMsgBuffer[40 + i] = (nanoseconds >> ((3-i)*8)) & 0xFF;
    }
    
    //Serial.printf("Sending DELAY_REQ (seq: %d) to %s\n", sequenceId, masterIP.toString().c_str());
    
    if(!UdpEvent.beginPacket(masterIP, localEventPort)) {
        Serial.println("Begin packet failed");
        return;
    }
    UdpEvent.write(ptpMsgBuffer, 44);
    if(UdpEvent.endPacket()) {
        lastPacketSent = millis();
        sequenceId++;
    } else {
        Serial.println("End packet failed");
    }
}

void handleSyncMessage(byte *buffer) {
  uint16_t recvSequence = (buffer[30] << 8) | buffer[31];
  if (recvSequence <= lastRecvSequence) {
    Serial.printf("Out of sequence SYNC: %d <= %d\n", recvSequence, lastRecvSequence);
    return;
  }
  
  lastRecvSequence = recvSequence;
  t1 = extractTimestamp(buffer, 34);
  t2 = getCurrentTimeInNanos();
  
  //Serial.printf("SYNC: seq=%d t1=%llu t2=%llu\n", recvSequence, t1, t2);
}

void handleFollowUpMessage(byte *buffer) {
  uint16_t recvSequence = (buffer[30] << 8) | buffer[31];
  t1 = extractTimestamp(buffer, 34);
  //Serial.printf("FOLLOW_UP: seq=%d precise t1=%llu\n", recvSequence, t1);
  calculateOffsetAndDelay();
}

void handleDelayRespMessage(byte *buffer) {
  uint16_t recvSequence = (buffer[30] << 8) | buffer[31];
  t4 = extractTimestamp(buffer, 34);
  //Serial.printf("DELAY_RESP: seq=%d t4=%llu\n", recvSequence, t4);
  calculateOffsetAndDelay();
}

void processPTPEventMessage(byte *buffer, int size) {
  byte messageType = buffer[0] & 0x0F;
  
  switch (messageType) {
    case SYNC:
      handleSyncMessage(buffer);
      break;
    default:
      Serial.printf("Unknown event message: 0x%02X\n", messageType);
      break;
  }
}

void processPTPGeneralMessage(byte *buffer, int size) {
  byte messageType = buffer[0] & 0x0F;
  
  switch (messageType) {
    case FOLLOW_UP:
      handleFollowUpMessage(buffer);
      break;
    case DELAY_RESP:
      handleDelayRespMessage(buffer);
      break;
    default:
      Serial.printf("Unknown general message: 0x%02X\n", messageType);
      break;
  }
}

void calculateOffsetAndDelay() {
    if (t1 && t2 && t3 && t4) {
        if (!isValidTimestamp(t1) || !isValidTimestamp(t2) ||
            !isValidTimestamp(t3) || !isValidTimestamp(t4)) {
            Serial.println("Invalid timestamps detected");
            t1 = t2 = t3 = t4 = 0;
            return;
        }

        int64_t master_to_slave = t2 - t1;
        int64_t slave_to_master = t3 - t4;

        offset = (master_to_slave + slave_to_master) / 2;
        roundTripDelay = (master_to_slave - slave_to_master) / 2;

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
  //Serial.printf("RTC adjusted by %lld ns\n", offset);
}

uint64_t extractTimestamp(byte *buffer, int startIndex) {
  uint64_t seconds = 0;
  for(int i = 0; i < 6; i++) {
    seconds = (seconds << 8) | buffer[startIndex + i];
  }
  
  uint32_t nanoseconds = 0;
  for(int i = 0; i < 4; i++) {
    nanoseconds = (nanoseconds << 8) | buffer[startIndex + 6 + i];
  }
  
  return (seconds * 1000000000ULL) + nanoseconds;
}

uint64_t getCurrentTimeInNanos() {
    if (!rtcInitialized) return 0;
    
    RTCDateTime now = rtc.getDateTime();
    return (uint64_t)now.unixtime * 1000000000ULL;
}

bool isValidTimestamp(uint64_t timestamp) {
    // Check if timestamp is within reasonable range (after year 2020)
    return timestamp > 1577836800000000000ULL; // 2020-01-01
}

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(10);
    }
    Serial.println("PTP Client Starting");
    
    WiFi.onEvent(WiFiEvent);
    
    // Initialize Ethernet
    ETH.begin();
    ETH.config(local_ip, gateway, subnet, dns);

    // Wait for Ethernet connection
    unsigned long startTime = millis();
    while (!ETH.linkUp()) {
        if (millis() - startTime > 10000) {
            Serial.println("Ethernet connection timeout");
            ESP.restart();
        }
        delay(500);
    }
    
    // Initialize I2C and RTC
    Wire.begin();
    rtc.begin();
    rtcInitialized = rtc.isReady();
    Serial.printf("RTC initialized: %d\n", rtcInitialized);
    rtc.setDateTime(2024, 11, 19, 12, 0, 0); // (year, month, day, hour, minute, second)
}

void loop() {
    static unsigned long lastDelayReq = 0;
    static unsigned long lastStatus = 0;
    static unsigned long lastAnnounce = 0;
    static bool networkActive = false;
    static unsigned long lastNetworkCheck = 0;
    static unsigned long lastValidSync = 0;
    
    // Network health check
    if (millis() - lastNetworkCheck > 5000) {
        if (!ethernetConnected || !udpInitialized) {
            Serial.println("Network connection issues detected");
            if (!ETH.linkUp()) {
                Serial.println("Ethernet link down, restarting...");
                ESP.restart();
            }
            initializeUDP();
        }
        lastNetworkCheck = millis();
    }

    // Process incoming packets
    int packetSize = UdpEvent.parsePacket();
    if (packetSize) {
        lastPacketReceived = millis();
        networkActive = true;
        byte messageType = ptpMsgBuffer[0] & 0x0F;
        if (messageType == SYNC) {
            lastValidSync = millis();
        }
        UdpEvent.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPEventMessage(ptpMsgBuffer, packetSize);
    }
    
    packetSize = UdpGeneral.parsePacket();
    if (packetSize) {
        lastPacketReceived = millis();
        networkActive = true;
        UdpGeneral.read(ptpMsgBuffer, PTP_MSG_SIZE);
        processPTPGeneralMessage(ptpMsgBuffer, packetSize);
    }

    // Check for network timeout
    if (millis() - lastPacketReceived > NETWORK_TIMEOUT && 
        millis() - lastPacketSent > NETWORK_TIMEOUT) {
        Serial.println("Network timeout, restarting UDP...");
        udpInitialized = false;
        initializeUDP();
    }

    // Send periodic DELAY_REQ
    if (millis() - lastDelayReq > 1000) {
        sendDelayReq();
        lastDelayReq = millis();
    }
    
    // Print status periodically
    /*if (millis() - lastStatus > 5000) {
        Serial.printf("Status - IP: %s Seq: %d Network: %s\n", 
            ETH.localIP().toString().c_str(), 
            sequenceId,
            networkActive ? "Active" : "Inactive");
        networkActive = false;
        lastStatus = millis();
    }*/

    // Watchdog for announce messages
    if (millis() - lastAnnounce > 30000) {
        Serial.println("No network activity, resetting...");
        ESP.restart();
    }

    // Only reset if no valid SYNC messages for 30 seconds
    if (millis() - lastValidSync > NETWORK_TIMEOUT) {
        Serial.println("No SYNC messages received, restarting...");
        ESP.restart();
    }
}

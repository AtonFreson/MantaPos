#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <DS3231.h>
#include <EEPROM.h> // Include EEPROM library
#include "esp_timer.h"

#define RTC_ISR_PIN 2

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

uint32_t storedSyncTime = 0;        // Variable to store the last sync time to EEPROM

unsigned long rtcUpdateStartTime = 0; // Track when RTC updates started
bool rtcUpdateExpired = false;        // Flag to indicate if RTC update period has expired
volatile uint64_t localSecondOffsetNs = esp_timer_get_time() * 1000ULL; // Offset in nanoseconds from the last full second
uint64_t shiftNs = esp_timer_get_time() * 1000ULL; // Shift in nanoseconds from the last full second

// Clock interrupt frequency
// 0x00 = 1 Hz
// 0x08 = 1.024 kHz
// 0x10 = 4.096 kHz
// 0x18 = 8.192 kHz (Default if frequency byte is out of range);
byte itr_freq = 0x00;

esp_timer_handle_t rtc_update_timer = NULL;

typedef struct {
    uint64_t newRtcTime;
} RtcUpdateArg;

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

void rtcSecondInterrupt () {
    localSecondOffsetNs = esp_timer_get_time() * 1000ULL;
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
    t2 = getCurrentTimeInNanos();
    
    lastRecvSequence = recvSequence;
}

void handleFollowUpMessage(byte *buffer) {
    t1 = extractTimestamp(buffer, 34);
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
    if (rtcUpdateExpired) return;

    // Start RTC update timer
    if (rtcUpdateStartTime == 0) {
        rtcUpdateStartTime = millis();
    }

    // Only update RTC for 30 seconds
    if (millis() - rtcUpdateStartTime < 30000) {
        // Calculate the corrected time
        /*uint64_t correctedTimeNs = t4;
        //uint64_t correctedTimeNs = t2 - offset;

        // Calculate the next full second in corrected time
        uint64_t nextFullSecondNs = ((correctedTimeNs / 1000000000ULL) + 1) * 1000000000ULL;

        // Calculate delay until the next full second in corrected time
        int64_t delayToNextSecondNs = nextFullSecondNs - t4;//t2;
        //int64_t delayToNextSecondNs = nextFullSecondNs - correctedTimeNs;

        if (delayToNextSecondNs < 0) {
            uint64_t offsetSeconds = -delayToNextSecondNs / 1000000000ULL + 1ULL;
            Serial.printf("Negative offset, adding %llu seconds\n", offsetSeconds);
            delayToNextSecondNs += offsetSeconds * 1000000000ULL;
        } else {
            uint64_t offsetSeconds = delayToNextSecondNs / 1000000000ULL;
            Serial.printf("Positive offset, removing %llu seconds\n", offsetSeconds);
            delayToNextSecondNs -= offsetSeconds * 1000000000ULL;
        }

        uint64_t delayUs = delayToNextSecondNs / 1000ULL;

        uint64_t newRtcTime = nextFullSecondNs / 1000000000ULL;

        scheduleRtcUpdate(delayUs, newRtcTime);*/



        //uint64_t correctedTimeNs = t2 - offset;
        uint64_t correctedTimeNs = t4 - offset/2;

        uint64_t nextFullSecondNs = ((correctedTimeNs / 1000000000ULL) + 1ULL) * 1000000000ULL;
        
        uint64_t delayToNextSecondNs = nextFullSecondNs - correctedTimeNs;
        uint64_t delayUs = delayToNextSecondNs / 1000ULL;
        uint64_t newRtcTime = nextFullSecondNs / 1000000000ULL;

        scheduleRtcUpdate(delayUs, newRtcTime);

        // Save time and offset to EEPROM
        storedSyncTime = newRtcTime;
        EEPROM.put(0, storedSyncTime); // Store time at address 0
        EEPROM.put(4, 0ULL); // Store offset at address 4
        EEPROM.commit();
        //Serial.println("Sync Time and Offset stored to EEPROM");
    } else {
        // Stop updating RTC after 30 seconds
        rtcUpdateExpired = true;
    }
}

void scheduleRtcUpdate(uint64_t delayUs, uint64_t newRtcTime) {
    if (rtc_update_timer != NULL) {
        // Timer already exists, delete it
        esp_timer_delete(rtc_update_timer);
        rtc_update_timer = NULL;
    }

    RtcUpdateArg* arg = (RtcUpdateArg*)malloc(sizeof(RtcUpdateArg));
    if (arg == NULL) {
        Serial.println("Failed to allocate memory for timer arg");
        return;
    }
    arg->newRtcTime = newRtcTime;

    esp_timer_create_args_t timer_args = {
        .callback = &rtcUpdateCallback,
        .arg = (void*)arg,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "rtc_update_timer"
    };

    esp_err_t err = esp_timer_create(&timer_args, &rtc_update_timer);
    if (err != ESP_OK) {
        Serial.printf("Failed to create timer: %s\n", esp_err_to_name(err));
        free(arg);
        return;
    }

    err = esp_timer_start_once(rtc_update_timer, delayUs);
    if (err != ESP_OK) {
        Serial.printf("Failed to start timer: %s\n", esp_err_to_name(err));
        esp_timer_delete(rtc_update_timer);
        rtc_update_timer = NULL;
        free(arg);
    } else {
        Serial.printf("RTC update scheduled in %llu us to set time to %llu... ", delayUs, newRtcTime);
    }
}

void rtcUpdateCallback(void* arg) {
    RtcUpdateArg* rtcArg = (RtcUpdateArg*)arg;
    uint64_t newRtcTime = rtcArg->newRtcTime;

    // Set the DS3231 RTC
    rtc.setEpoch(newRtcTime, false);

    Serial.printf("RTC updated to %llu, locally timestamped at %llu ns\n", newRtcTime, localSecondOffsetNs);

    // Delete the timer
    if (rtc_update_timer != NULL) {
        esp_timer_delete(rtc_update_timer);
        rtc_update_timer = NULL;
    }

    // Free the arg
    free(rtcArg);
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

// Use esp_timer_get_time() to get the current time from ESP32 in microseconds since boot
uint64_t getCurrentTimeInNanos() {
    // Get current time from the RTC
    DateTime now = RTClib::now();
    noInterrupts();
    shiftNs = esp_timer_get_time() * 1000ULL - localSecondOffsetNs;
    interrupts();
    // Serial.printf("Local offset: %llu ns. Local RTC set time: %llu ns\n", localOffset, localRTCSetTimeNs);

    return (((uint64_t)now.unixtime() * 1000000000ULL) + shiftNs); // Convert to nanoseconds (x9)
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
    rtc.enableOscillator(true, false, itr_freq); // enable the 1 Hz output on interrupt pin

    // set up to handle interrupt from 1 Hz pin
    pinMode (RTC_ISR_PIN, INPUT_PULLUP);
    attachInterrupt (digitalPinToInterrupt (RTC_ISR_PIN), rtcSecondInterrupt, FALLING);

    EEPROM.begin(512); // Initialize EEPROM with size 512 bytes
    EEPROM.get(0, storedSyncTime); // Read the last stored sync time from EEPROM
    EEPROM.get(4, offset); // Read the stored offset from EEPROM
}

void loop() {
    if (rtcUpdateExpired) {
        Serial.println("RTC update period has expired, waiting for reset...");
        EEPROM.get(0, storedSyncTime);
        EEPROM.get(4, offset);
        Serial.printf("Final stored sync time: %u\n", storedSyncTime);
        Serial.printf("Final stored offset: %lld\n", offset);
        while (true) {
            // Do nothing
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

    if (millis() - lastPacketSent > 2000) sendDelayReq();
}

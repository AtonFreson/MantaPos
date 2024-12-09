#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <DS3231.h>
#include <ESP1588.h>
#include <EEPROM.h>

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

int lastPrint = 0;

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
            break;
        case ARDUINO_EVENT_ETH_STOP:
            Serial.println("ETH: Stopped");
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
        Serial.println("UDP fully initialized");
    } else {
        Serial.println("UDP initialization failed");
    }
}

void PrintPTPInfo(ESP1588_Tracker & t) {
	const PTP_ANNOUNCE_MESSAGE & msg=t.GetAnnounceMessage();
	const PTP_PORTID & pid=t.GetPortIdentifier();

	Serial.printf("    %s: ID ",t.IsMaster()?"Master":"Candidate");
	for(int i=0;i<(int) (sizeof(pid.clockId)/sizeof(pid.clockId[0]));i++) {
		Serial.printf("%02x ",pid.clockId[i]);
	}

	Serial.printf(" Prio %3d ",msg.grandmasterPriority1);

	Serial.printf(" %i-step",t.IsTwoStep()?2:1);

	Serial.printf("\n");
}


void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("PTP RTC Sync Starting");

    WiFi.onEvent(WiFiEvent);
    ETH.begin();
    ETH.config(local_ip, gateway, subnet, dns);

    Wire.begin();
    rtc.setClockMode(false); // Set 24h format

    EEPROM.begin(512); // Initialize EEPROM with size 512 bytes

    esp1588.SetDomain(0);	//the domain of your PTP clock, 0 - 31
    esp1588.Begin();
}

void loop() {
    esp1588.Loop();	//this needs to be called OFTEN, at least several times per second but more is better. forget controlling program flow with delay() in your code.
    ESP1588_Tracker & m=esp1588.GetMaster();

    // print a status message every four seconds
    if (millis() - lastPrint > 4000) {
        lastPrint = millis();

        Serial.printf("PTP status: %s   Master %s   Delay %s\n", esp1588.GetLockStatus()?"LOCKED":"UNLOCKED", m.Healthy()?"OK":"no", esp1588.GetShortStatusString());
        PrintPTPInfo(m);
    }

    uint64_t currentMillis = esp1588.GetEpochMillis64();
    if (esp1588.GetLockStatus() && m.Healthy() && esp1588.GetLastDiffMs() == 0 && currentMillis%1000 == 0) {
        rtc.setEpoch(currentMillis/1000, false);

        Serial.printf("PTP status: %s   Master %s   Delay %s\n", esp1588.GetLockStatus()?"LOCKED":"UNLOCKED", m.Healthy()?"OK":"NOT OK", esp1588.GetShortStatusString());
        PrintPTPInfo(m);
        
        uint64_t offset = 0;

        Serial.printf("\nRTC set. Saving unixtime as %lu and offset as %llu\n", currentMillis/1000, offset);
        EEPROM.put(0, currentMillis/1000); // Store time at address 0
        EEPROM.put(4, offset); // Store offset at address 4
        EEPROM.commit();

        Serial.println("Stopping... \n");
        while(true) {
            // Do nothing
        }
    }
}
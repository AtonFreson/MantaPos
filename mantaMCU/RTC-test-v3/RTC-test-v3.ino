#include <ETH.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <DS3231.h>
#include <ESP1588.h>
#include <EEPROM.h>

DS3231 rtc;


bool ethernetConnected = false;
bool udpInitialized = false;


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


void setup()
{
  Serial.begin(115200);

  // The WiFi part is borrowed from the ESP8266WiFi example.

  // We start by connecting to a WiFi network

  Serial.println();
  Serial.println();


  WiFi.onEvent(WiFiEvent);
  ETH.begin();
  ETH.config(local_ip, gateway, subnet, dns);

  Wire.begin();
  rtc.setClockMode(false); // Set 24h format

  EEPROM.begin(512); // Initialize EEPROM with size 512 bytes

  esp1588.SetDomain(0);	//the domain of your PTP clock, 0 - 31
  esp1588.Begin();
}

void loop()
{

  esp1588.Loop();	//this needs to be called OFTEN, at least several times per second but more is better. forget controlling program flow with delay() in your code.

  
  static uint32_t last_millis=0;

  if(((esp1588.GetMillis()+250) / 4000) != ((last_millis+250) / 4000))	//print a status message every four seconds, slightly out of sync with the LEDs blinking for improved blink accuracy.
  {
    last_millis=esp1588.GetMillis();

    ESP1588_Tracker & m=esp1588.GetMaster();
    ESP1588_Tracker & c=esp1588.GetCandidate();


    Serial.printf("PTP status: %s   Master %s, Candidate %s\n",esp1588.GetLockStatus()?"LOCKED":"UNLOCKED",m.Healthy()?"OK":"no",c.Healthy()?"OK":"no");

    //this function is defined below, prints out the master and candidate clock IDs and some other info.
    PrintPTPInfo(m);
    PrintPTPInfo(c);

    Serial.println(esp1588.GetShortStatusString());
    Serial.println(esp1588.GetLastDiffMs());
    
    Serial.printf("\n");

  }
  
  uint64_t currentMillis = esp1588.GetEpochMillis64();
  if (esp1588.GetLockStatus() && esp1588.GetLastDiffMs() == 0 && currentMillis%1000 == 0) {
    rtc.setEpoch(currentMillis/1000, false);

    EEPROM.put(0, currentMillis/1000); // Store time at address 0
    EEPROM.put(4, 0ULL); // Store offset at address 4
    EEPROM.commit();

    Serial.println("RTC set. Stopping...");
    while(true) {
      // Do nothing
    }
  }
	
}

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
    } else {
        udpInitialized = false;
        Serial.println("UDP initialization failed");
    }
}

void PrintPTPInfo(ESP1588_Tracker & t)
{
	const PTP_ANNOUNCE_MESSAGE & msg=t.GetAnnounceMessage();
	const PTP_PORTID & pid=t.GetPortIdentifier();

	Serial.printf("    %s: ID ",t.IsMaster()?"Master   ":"Candidate");
	for(int i=0;i<(int) (sizeof(pid.clockId)/sizeof(pid.clockId[0]));i++)
	{
		Serial.printf("%02x ",pid.clockId[i]);
	}

	Serial.printf(" Prio %3d ",msg.grandmasterPriority1);

	Serial.printf(" %i-step",t.IsTwoStep()?2:1);

	Serial.printf("\n");
}


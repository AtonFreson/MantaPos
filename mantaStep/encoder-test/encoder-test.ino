// Pin definitions for SCA50 differential signals
#define ENCODER_PIN_A_POS 13  // A
#define ENCODER_PIN_A_NEG 14  // A-
#define ENCODER_PIN_B_POS 32  // B
#define ENCODER_PIN_B_NEG 33  // B-
#define INDEX_PIN_Z_POS 5     // Z
#define INDEX_PIN_Z_NEG 4     // Z-
 
// Configuration
#define PPR 400  // Pulses Per Revolution
#define COUNTS_PER_REV (PPR * 4)  // In quadrature mode
#define WHEEL_DIAMETER_MM 82.53
#define DISTANCE_PER_REV (WHEEL_DIAMETER_MM * PI / 1000.0)  // meters per revolution
#define DISTANCE_PER_COUNT (DISTANCE_PER_REV / COUNTS_PER_REV)

volatile long encoderCount = 0;
long lastEncoderCount = 0;
unsigned long lastPrintTime = 0;
const unsigned long printInterval = 10; // 100ms update rate

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
        // Optional: Implement index pulse handling
        // For example, you could use this to verify position or reset count
    }
}

void setup() {
    Serial.begin(115200);
    
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
    
    Serial.println("\nSCA50-400 Encoder Test");
    Serial.println("PPR: 400");
    Serial.println("Resolution: ±0.10mm\n");
}

void loop() {
    unsigned long currentTime = millis();
    
    // Check for serial input
    if (Serial.available() > 0) {
        String input = Serial.readString();
        input.trim(); // Remove any leading/trailing whitespace
        Serial.println("Received input: " + input);
        if (input.equals("reset")) {
            noInterrupts();
            encoderCount = 0;
            interrupts();
            Serial.println("Encoder count reset.");
        }
    }

    if (currentTime - lastPrintTime >= printInterval) {
        // Calculate speed and position only if count changed
        if (encoderCount != lastEncoderCount) {
            noInterrupts();  // Disable interrupts while reading volatile
            long currentCount = encoderCount;
            interrupts();
            
            float revolutions = (float)currentCount / COUNTS_PER_REV;
            float distance = currentCount * DISTANCE_PER_COUNT;
            
            // Calculate speed
            float countsPerSec = (float)(currentCount - lastEncoderCount) * (1000.0 / printInterval);
            float rpm = (countsPerSec * 60.0) / COUNTS_PER_REV;
            float speed_m_per_s = countsPerSec * DISTANCE_PER_COUNT;
            char buffer[100];
            snprintf(buffer, sizeof(buffer), 
                    "Rev: %8.3f      RPM: %8.3f      Speed (m/s): %8.3f      Dist (m): %9.4f", 
                    revolutions, rpm, speed_m_per_s, distance);
            Serial.println(buffer);
            
            lastEncoderCount = currentCount;
        }
        lastPrintTime = currentTime;
    }
}
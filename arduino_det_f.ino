#include <Stepper.h>

const int stepsPerRevolution = 2048;

Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);

  myStepper.setSpeed(10); 
  
  Serial.println("SYSTEM_READY");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'N') {
      myStepper.step(256);
      delay(200);
      Serial.println("PICTURE_READY");
      digitalWrite(8, LOW);
      digitalWrite(9, LOW);
      digitalWrite(10, LOW);
      digitalWrite(11, LOW);
    }
  }
}
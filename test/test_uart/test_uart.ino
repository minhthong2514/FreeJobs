int count = 1;
String data;

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    // Read data
    data = Serial.readStringUntil('\n');
    Serial.println(data);
  }

  // Send data
  Serial.println(count);
  count++;
  delay(1000);
}

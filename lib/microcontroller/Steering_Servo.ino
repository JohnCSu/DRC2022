/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 https://www.arduino.cc/en/Tutorial/LibraryExamples/Sweep
*/



/*
  The servo Controller is set at 45 degrees to be straight. +- 45 is for turning

  0-44 degrees: Turn Right
  46-90 degrees: Turn Left 
 */
#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position

void setup() {
  myservo.attach(3);  // attaches the servo on pin 9 to the servo object
  myservo.write(45);
//  for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
//  // in steps of 1 degree
//    myservo.write(pos);
//    delay(15);        
//  }
//  delay(1000);
//  for (pos = 180; pos >=0; pos -= 1) { // goes from 0 degrees to 180 degrees
//  // in steps of 1 degree
//    myservo.write(pos);
//    delay(15);        
//  }
}

void loop() {

  delay(1000);
  
//  for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
//    // in steps of 1 degree
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
////    delay(15);                       // waits 15 ms for the servo to reach the position
//  }
//  for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(15);                       // waits 15 ms for the servo to reach the position
//  }
}

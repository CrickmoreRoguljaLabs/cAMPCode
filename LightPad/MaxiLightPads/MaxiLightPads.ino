/* MAXILIGHTPADS

If someone pushes a button, turn on the light pad. Then wait 20 minutes.
After that, turn off the light for 30 seconds.
20 minutes later, turn off the light for 90 seconds.

Adjust as following:

timeOfFirstWindow = time between button push and the first window
firstWindow = duration of the first window
timeOfSecondWindow = time between BUTTON PUSH and the second window
secondWindow = duration of the second window

SCT 04/07/2018. A sad time.
*/

const int timeOfFirstWindow = 15; // in minutes
const int firstWindow = 30; // in seconds
const int timeOfSecondWindow = 30; // in minutes
const int secondWindow = 90; // in seconds
const int timeOfThirdWindow = 60;
const int timeOfFourthWindow = 90;

const int numPins = 12;

unsigned long startTime = 0;

int inOutMap[numPins] = {36,37,38,39,40,41,42,43,44,45,46,47};
int inPins[numPins] =   {24,25,26,27,28,29,30,31,32,33,34,35};

boolean activated[numPins] = {false, false, false, false, false, false, false, false, false, false, false};
unsigned long buttonTime[numPins] = {0,0,0,0,0,0,0,0,0,0,0};

void setup() {
// Set up the output pins and stuff
  Serial.begin(9600);
  for(int k = 0; k < numPins; k++){
   pinMode(inOutMap[k] , OUTPUT);
   pinMode(inPins[k], INPUT);
   digitalWrite(inOutMap[k], HIGH); // god this feels good. So sick of BuckPucks
  }
  startTime = millis();
}

// And now you wait for someone to push a button
void loop() {
  unsigned long currentTime = millis(); // Current time.
  // check all the input pins to see if the button is pushed
  for(int inScan = 0 ; inScan < numPins; inScan++) {
    if (digitalRead(inPins[inScan]) == HIGH) {
     Serial.println( String(inPins[inScan]) );
     if (activated[inScan] == false) {
       activated[inScan] = true;
       buttonTime[inScan] = currentTime;  
     }
     // Ok so now if the switch is on, check to see if the light should be on
     if (activated[inScan] == true) {
       // if we haven't hit the first window yet, keep everything on
       if ( ((currentTime - buttonTime[inScan])/(60000)) < timeOfFirstWindow) {
         digitalWrite(inOutMap[inScan], HIGH);
       }
       // check if we're still in the first window
       else if( ((currentTime - (buttonTime[inScan] + 60000*timeOfFirstWindow))/(1000)) < firstWindow) {
         digitalWrite(inOutMap[inScan], LOW);
       }
       // check if it's the second window yet
       else if( ((currentTime - buttonTime[inScan])/(60000)) < timeOfSecondWindow ) {
         digitalWrite(inOutMap[inScan], HIGH);
       }
       // now we're in the second window
       else if( ((currentTime - (buttonTime[inScan] + 60000*timeOfSecondWindow))/(1000)) < secondWindow) {
         digitalWrite(inOutMap[inScan], LOW);
       }
        // check if it's the third window yet
       else if( ((currentTime - buttonTime[inScan])/(60000)) < timeOfThirdWindow ) {
         digitalWrite(inOutMap[inScan], HIGH);
       }
       // now we're in the third window
       else if( ((currentTime - (buttonTime[inScan] + 60000*timeOfThirdWindow))/(1000)) < firstWindow) {
         digitalWrite(inOutMap[inScan], LOW);
       }
        // check if it's the fourth window yet
       else if( ((currentTime - buttonTime[inScan])/(60000)) < timeOfFourthWindow ) {
         digitalWrite(inOutMap[inScan], HIGH);
       }
       // now we're in the fourth window
       else if( ((currentTime - (buttonTime[inScan] + 60000*timeOfFourthWindow))/(1000)) < secondWindow) {
         digitalWrite(inOutMap[inScan], LOW);
       }
       else {
         digitalWrite(inOutMap[inScan],HIGH);
       }
      }
    }
    else {
      digitalWrite(inOutMap[inScan], LOW);
      activated[inScan] = false;
    }
  }
}

from sys import byteorder
import serial
import time
from ast import literal_eval

def clamp(x,maxval = 255, minval = 0):
    if x <= minval:
        return minval
    elif x >= maxval:
        return maxval

    return x

class arduino():

    def __init__(self, port = 'COM5'):
        self.arduino = serial.Serial(port = port, baudrate = 9600, timeout = .1)
        self.start()
    def start(self):
        start = int.to_bytes(200,length=1,byteorder='big')
        setup = True
        while(setup):
            bs = self.readData()
            if bs == start:
                print('Start')
                return
            time.sleep(0.01)

    def sendData(self,angle=0,speed = 20,state = 0):
        '''
        Send 5 bytes at a time:

        0: 245 starting byte
        1: Steering (0-90) will be   
        2: speed/servo set (0-60) maybe have 0-20 as reverse, 21-60 for speed control
        3: state (undecided) default will be 0 (stop all )
        4: 250 ending
        
        '''
        angle = int(clamp(45+angle,90,0))
        # arduino = serial.Serial(port = 'COM5', baudrate = 9600, timeout = .1)
        # arduino.write(bytes(data , 'utf-8'))
        self.arduino.write(bytearray([245,angle,speed,state,250]))
        
    def readData(self):
        #Send the same byte packet back
        return (self.arduino.read(5))
if __name__ == '__main__':

    ask = True
    a = arduino()
    while(1):
        if ask:
            data = input('Enter The data as a list [1,2,3]:\t')
            data = literal_eval(data)
            a.sendData(*data)
            time.sleep(0.1)
            by = a.readData()
            print([b for b in by])
                # print(type(b))
            
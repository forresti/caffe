
import re

#dict of computation times for each type of CNN layer
def initDict(layerTypes):
    myDict = dict()
    for layer in layerTypes:
        myDict[layer] = 0.0 #start with 0 sec ... will increment time later
    return myDict

def parseProfile(fname):

    layerTypes = ['conv', 'fc', 'pool', 'relu', 'drop', 'pad', 'data'] 
    timingDict = initDict(layerTypes)

    lines = open(fname).read().splitlines()

    for l in lines:
        #e.g. E1225 15:46:51.223964 21781 net_speed_benchmark.cpp:49] [SPLIT HERE] drop6   forward: 0.32 seconds.
        lineParts = l.split(']') #lineParts[0] = time/code line, lineParts[1] = drop6 forward: 0.32 seconds
        line = lineParts[1] #ignore the time and code line

        #these printouts only have 1 number after the 'time/code line' stuff
        numbers = re.findall(r'\d+.\d+', line) #find all numbers that have a decimal point. (e.g. 0.32).

        #note: sometimes, layers take 0 seconds, which shows with no decimal point. this is fine... 
        #      no need to have an additional special case for this.
        #if numbers is not None:
        if type(numbers) is list and len(numbers) > 0:
            compTime = float(numbers[0])
        else:
            compTime = None

        #figure out this string's layer type (e.g. conv, pool, etc)
        myLayer = None
        for layer in layerTypes:
            if layer in line:
                myLayer = layer

        if myLayer != None and compTime != None:
            timingDict[myLayer] = timingDict[myLayer] + compTime

    return timingDict

fname = 'forrest_profile_orig.txt' #unmodified caffe Toronto-style net

timingDict = parseProfile(fname)
print timingDict



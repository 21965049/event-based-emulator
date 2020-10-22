# Event-based camera emulator
# By Vladislav Matveev
# 2020

import os
import time
import numpy as np
import cv2

# 'q' is the default exit key

# ~~~~~~~~~~~~~~~~~~~~~~~~~~SETUP AND INPUT PARAMETERS~~~~~~~~~~~~~~~~~~~~~~~~~~
downscale_factor=4 #downscaling resolutions of image and video objects for faster processing, set to 1 for native resolution
scene = True  # whether to draw the reference values or not, to see change need to find updates in which the change happens
              # otherwise the output will be black for all updates in which there are no spikes
              # takes 11 updates to run out of spike-able pixels at default settings for the 'lena.bmp' image
filename = "lena.bmp" #image filepath, ignored if video is true
save_final = False #saves file, only for images
video = False  #attempts to read from webcam/video file
video_src = "videoplayback.mp4" #video filepath, ignored if webcam is true
webcam = False #sets webcam on as source under 'video'
webcam_src = 0 #sets the webcam source, increase value if it is the wrong webcam

# ~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN PARAMETERS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
exposure_time = 0.1   # aka integration time, or timeslot when pixels can 'charge' and spike
charge_rate = 1  # how much a pixel should be charged by, per time interval, as an integer value
charge_speed = 0.0005  # time in seconds for every increment of charge, 0.000392 is approx 1:1 charge relationship with default settings
event_thres = 15  # pixel integer value increase or decrease that would trigger an event
arbitrate = False  # set whether there should be arbitration, affects performance, lowers frame update output
arbiter_type = "FIFO"  # select arbitration scheme: "FIFO", "RAND", "FAIR"
delta_thres = 0.0001  # amount of time difference allowed between events before arbitration stops, timespan = earliest spike in the update+delta_thres
fps = 30  # frames to generate per second
frame_time = 1/fps  # the amount of time the emulator is allowed to process a frame for, can output multiple updates over this time with lighter loads
update_limit = 0  # the amount of updates to frames allowed per frame time, 0 for no limit, 1 = pixels can spike once to event_thres and no more
                  # change this value accordingly to showcase arbitration outputs, the emulator works too fast to see a difference in the final output

# Defaults
# exposure_time = 0.1
# charge_rate = 1
# charge_speed = 0.0005
# event_thres = 15
# arbitrate = False
# arbiter_type = "FIFO"
# delta_thres = 0.0001
# fps = 30 
# frame_time = 1/fps
# update_limit = 0

def is_file(filename):
    if os.path.isfile(filename):
        print("File found.")
        return True
    else:
        print("File not found or does not exist.")
        return False


def check_img(imgname):
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print("Image is of a valid format.")
        return img
    else:
        print("The file is not a valid image format.")
        return None

class arbiter:
    global priorities
    def __init__(self, spikes, acks, thres):
        self.spikes=spikes
        self.acks=acks
        self.prio=priorities
        self.thres=thres
        self.start=0

    #Keep track of time per frame update to drop requests outside of timeframe
    def check_time(self):
        if frame_time/2<time.time()-self.start:
            return True
        else:
            return False

    #Acknowledge
    def ack(self, index):
        self.acks[index]=True

    def process(self, time, timespan):
        arr= np.where(self.spikes >= timespan, 0, self.spikes)
        arr = np.where(self.spikes <= time, 0, arr)
        return arr

    def check_done(self):
        return np.all(self.acks[self.spikes>0])

    #FIFO
    def arbi_fixed(self):
        #arbitime=time.time()
        if not self.spikes[self.spikes > 0].size==0:
            self.start = np.min(self.spikes[self.spikes > 0])
            self.acks = np.where(self.spikes<=self.start+self.thres, True, False)
            #arbitime2=time.time()
            #print("total time taken "+str(arbitime2-arbitime)+"sec")
        return self.acks

    #Randomly switch requests
    def arbi_rand(self):
        rand=np.random.random()/1000 #generate a random value for the current ack update
        if not self.spikes[self.spikes > 0].size == 0:
            self.start = np.min(self.spikes[self.spikes > 0])
            self.start+=rand
            self.acks = np.where(self.spikes <= self.start + self.thres, True, False)
        return self.acks

    #Balance priorities
    def arbi_fair(self):
        global priorities
        np.seterr(divide='ignore', invalid='ignore') #ignore div by 0 since the values would not be evaluated
        new_prio = np.where(self.spikes > 0, 1/self.spikes, 0).astype("double")
        if not new_prio[new_prio>0].size==0:
            min_prio = np.min(new_prio[new_prio > 0])
            update_prio = self.prio+new_prio
            self.start = np.min(update_prio[update_prio > 0])
            self.acks = np.where(update_prio <= self.start+min_prio, True, False)
            priorities=update_prio
        return self.acks


def setup(data):
    global ref_array
    np.seterr(divide='ignore', invalid='ignore') #ignore div by 0 since the values would not be evaluated
    replacement=ref_array if scene else 0 #set whether reference scene is drawn or not
    # Charge equation:
    # rate*data/255 -> scaling charge rate according to pixel's max charge value (smaller max=less charge rate)
    # time/speed -> how many charge 'cycles' will be available with the given exposure time and charge speed
    # cycles*scaled_rate -> final integer value that the pixel reaches
    #final_val = charge_cycles * scaled_rate
    scaled_charge=np.floor((charge_rate*data.astype("int16")/255)*(exposure_time/charge_speed)).astype("int16")
    ref_spikes=np.where(scaled_charge>0, ref_array, 0)
    scaled_diff=(scaled_charge-ref_spikes.astype("int16"))
    over_scaled_thres=np.abs(scaled_diff)>=event_thres
    polarity=scaled_diff>0 #storing polarity of the pixel change after scaling
    spike_data=np.where(polarity, ref_spikes.astype("int16")+event_thres, ref_spikes.astype("int16")-event_thres)
    spike_data=np.where(spike_data/255>1, 255, spike_data)#.astype("uint8")
    spike_data=np.where(over_scaled_thres, spike_data, 0).astype("uint8")
    #Calculate spike time data and pass it into arbiter emulation
    arbi_data=np.where(over_scaled_thres, (charge_speed * event_thres / charge_rate * data / 255), 0)
    #Call to arbitrate
    acks=emulate_arbiters(arbi_data, arbiter_type)
    final_array=np.where(over_scaled_thres & acks, spike_data, replacement) # pixels that don't spike are replaced with the selected replacement
    ref_array=np.where(over_scaled_thres, spike_data, ref_array)
    return final_array

def frame(img):
    global start
    start = time.time()
    time_spent = 0
    update_count = 0
    while not time_spent > frame_time:
        #print("frame: "+str(update_count))
        if update_limit is not 0 and update_count > update_limit:
            break
        # frame data
        final = setup(img)
        #can uncomment these and increase frame time to 100+ to see frame-by-frame changes
        #cv2.imshow('temp',final)
        #cv2.waitKey()
        finish = time.time()
        update_count += 1
        time_spent = finish - start
    #print("total updates: "+str(update_count))
    #print("total time for updates: "+str(time_spent))
    return final

def emulate_arbiters(spikes, arbi_type):
    #Set arbiter logic here
    global img_size
    if arbitrate:
        acks=np.zeros(img_size[::-1], dtype="bool") #reinitialise acks as pixels are reset
        arbi = arbiter(spikes, acks, delta_thres)
        if arbi_type=="FIFO":
            acks=arbi.arbi_fixed()
        elif arbi_type=="RAND":
            acks=arbi.arbi_rand()
        elif arbi_type=="FAIR":
            acks=arbi.arbi_fair()
    else:
        acks=np.ones(img_size[::-1], dtype='bool') #set acks to all true if no arbitration takes place
    return acks

def displayimg(name, data, winsize, move):
    size=img_size[0]*winsize
    cv2.imshow(name, data)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size, size)
    cv2.moveWindow(name, move[0], move[1])

def img():
    global img_size, final_array, ref_array, priorities
    ogs = time.time()
    factor=2 #used for increasing opencv window sizes
    imgr = check_img(filename)
    img_size = tuple([int(i / downscale_factor) for i in imgr.shape])
    img = cv2.resize(imgr, img_size, interpolation=cv2.INTER_AREA)
    final_array = np.zeros(img_size[::-1], dtype="uint8")  # constructing final image/s
    ref_array = np.zeros(img_size[::-1], dtype="uint8")  # reference array, previous frame/old value
    #Cumulative priority measurement
    priorities=np.zeros(img_size[::-1], dtype="double")
    displayimg('initial', img, factor, (500, 400))
    #print("file & img check takes: " + str(time.time() - ogs) + "sec")
    final = frame(img)
    ogf = time.time()
    print("Actual exec time= " + str(ogf - ogs) + "sec")
    displayimg('output', final, factor, (628, 400))
    cv2.waitKey()
    cv2.destroyAllWindows()
    if save_final:
        cv2.imwrite("output.bmp", final)

class get_frame:
    def __init__(self, cap):
        self.stopped=False
        self.cap=cap
        self.ret, self.frame=self.cap.read()

    def ret_self(self):
        return self

    def next_frame(self):
        if not self.ret:
            self.stopped=True
        else:
            self.ret, self.frame=self.cap.read()
            return self.frame

def video_out():
    global img_size, final_array, ref_array, priorities
    #ogs = time.time()
    #factor=2 #used for increasing opencv window sizes
    if not webcam:
        src = video_src
        cap = cv2.VideoCapture(src)
    else:
        src=webcam_src
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if not(cap.isOpened()):
        cap.open(src)
    img_size = tuple([int(i / downscale_factor) for i in (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    frames=get_frame(cap).ret_self()
    final_array = np.zeros(img_size[::-1], dtype="uint8")  # constructing final image/s
    ref_array = np.zeros(img_size[::-1], dtype="uint8")  # reference array, previous frame/old value
    # Cumulative priority measurement
    priorities = np.zeros(img_size[::-1], dtype="double")
    while True:
        imgr=frames.next_frame()
        if imgr is not None:
            imgg=cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(imgg, img_size, interpolation=cv2.INTER_AREA)
            frame_img=frame(img)
            cv2.imshow('frame', frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q') or frames.stopped:
            break
    #ogf = time.time()
    #print("Actual exec time= " + str(ogf - ogs) + "sec")
    cap.release()
    cv2.destroyAllWindows()

def main():
    # filename=input("Enter image name and extension (same directory):")
    if not video:
        if is_file(filename):
            img()
    else:
        #set up webcam/video readout
        if is_file(filename):
            video_out()


if __name__ == "__main__":
    main()

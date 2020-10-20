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
#pix_array = np.ndarray(img_size, dtype="object_")
scene = True  # whether to draw the reference values or not, to see change need to find updates in which the change happens
              # otherwise the output will be black for all updates in which there are no spikes
              # takes 11 updates to run out of spike-able pixels at default settings
filename = "lena.bmp" #image filepath, ignored if video is true
video = False  #attempts to read from webcam/video file
video_src = "videoplayback.mp4" #video filepath, ignored if webcam is true
webcam = False #sets webcam on as source under 'video'
webcam_src = 0 #sets the webcam source, increase value if it is the wrong webcam

# ~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN PARAMETERS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
exposure_time = 0.1   # aka integration time, or timeslot when pixels can 'charge' and spike
charge_rate = 1  # how much a pixel should be charged by, per time interval, as an integer value
charge_speed = 0.0005  # time in seconds for every increment of charge, 0.000392 is approx 1:1 charge relationship with default settings
event_thres = 15  # pixel integer value increase or decrease that would trigger an event
arbitrate = True  # set whether there should be arbitration, affects performance, lowers frame update output
arbiter_type = "FIFO"  # select arbitration scheme: "FIFO", "RAND", "FAIR"
delta_thres = 0.05  # amount of time difference allowed between events before arbitration takes place
fps = 30  # frames to generate per second
frame_time = 1/fps  # the amount of time the emulator is allowed to process a frame for, can output multiple updates over this time with lighter loads
#OR define frame time exactly for testing arbitration outputs, comment the line above and uncomment the line below
#frame_time = 0.010
update_limit = 0  # the amount of updates to frames allowed per frame time, 0 for no limit, 1 = pixels can spike once to event_thres and no more


# Defaults
# exposure_time = 0.100
# charge_rate = 1
# charge_speed = 0.0005
# event_thres = 15
# arbitrate = False
# arbiter_type = "FIXED"
# delta_thres = 1
# fps = 30  # frames to generate per second
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
    global priorities, acks, start
    def __init__(self, spikes, thres):
        self.spikes_og=spikes
        self.spike_int=np.argsort(spikes, axis=None)
        self.spikes=np.copy(spikes.flatten()[self.spike_int])
        self.thres=thres
        self.late=0

    #Keep track of time per frame update to drop requests outside of timeframe
    def check_time(self):
        if frame_time<time.time()-start:
            return True
        else:
            return False

    #Acknowledge
    def ack(self, pix):
        if pix is not None:
            acks[np.where(self.spikes==pix)]=True

    def process(self, time, timespan):
        print(timespan)
        print(np.where(self.spikes<=timespan))
        return np.where(self.spikes<=timespan)

    #FIFO
    def arbi_fixed(self):
        i=0
        for event in self.spikes:
            event_pos=self.spike_int
            time=self.spikes[i]
            span=time+self.thres
            events=self.process(time, span)
            print("here")
            print(self.spikes[events])
            print(self.spikes[events].shape[0])
            if i+1 < self.spikes[events].shape[0]:
                next_event=self.spikes[i+1]
                if event <= next_event:
                    print("event")
                    print(event)
                    print("next")
                    print(next_event)
                    self.ack(event)
                    self.late = next_event
                else:
                    self.ack(next_event)
                    self.late = event
            i+=1
            if self.check_time():
                break

    #Randomly switch requests
    def arbi_rand(self):
        rand=np.random.random()
        print(rand)
        if rand>0.5:
            self.ack(pix1)
            self.late = pix2
        else:
            self.ack(pix2)
            self.late = pix1

    #Balance priorities
    def arbi_fair(self):
        if priorities[pix1]<priorities[pix2]:
            self.ack(pix1)
            priorities[pix1]=priorities[pix1]+1/self.spikes[pix1]
            self.late = pix2
        else:
            self.ack(pix2)
            self.late = pix1

def setup(data):
    global ref_array, acks
    diff=(data.astype("int16")-ref_array.astype("int16")) #storing difference as a signed 16-bit integer to allow -ve values
    over_thres=np.abs(diff)>=event_thres #boolean array for pixels that spiked
    data=np.where(over_thres,data,0)
    replacement=ref_array if scene else 0 #set whether reference scene is drawn or not
    # Charge equation:
    # rate*data/255 -> scaling charge rate according to pixel's max charge value (smaller max=less charge rate)
    #scaled_rate = charge_rate * self.max / 255
    # time/speed -> how many charge 'cycles' will be available with the given exposure time and charge speed
    #charge_cycles = exposure_time / charge_speed
    # cycles*scaled_rate -> final integer value that the pixel reaches
    #final_val = charge_cycles * scaled_rate
    scaled_charge=np.round((charge_rate*data/255)*(exposure_time/charge_speed)).astype("uint8")
    # include polarity
    # find scaled diff and see which pixels did actually spike
    # set spiked pix values to pix+-thres, remember uint8
    ref_spikes=np.where(scaled_charge>0, ref_array, 0)
    scaled_diff=(scaled_charge.astype("int16")-ref_spikes.astype("int16"))
    over_scaled_thres = np.logical_and(np.abs(scaled_diff) >= event_thres, over_thres)
    polarity=scaled_diff>0 #storing polarity of the pixel change after scaling
    spike_data=np.where(polarity, ref_spikes+event_thres, ref_spikes-event_thres)
    spike_data=np.where(over_scaled_thres, spike_data, 0)
    #Calculate spike time data and pass it into arbiter emulation
    arbi_data=np.where(over_scaled_thres, (charge_speed * event_thres / charge_rate * data / 255), 0)
    emulate_arbiters(arbi_data, arbiter_type)
    final_array=np.where(over_scaled_thres & acks, spike_data, replacement) # pixels that don't spike are replaced with the selected replacement
    ref_array=np.where(spike_data>0, spike_data, ref_array)
    return final_array

def frame(img):
    global start
    start = time.time()
    time_spent = 0
    update_count = 0
    while not time_spent > frame_time:
        # print("frame: "+str(frame_count))
        if update_limit is not 0 and update_count > update_limit:
            break
        # frame data
        final = setup(img)
        finish = time.time()
        update_count += 1
        time_spent = finish - start
    #print("total updates: "+str(update_count))
    #print("total time for updates: "+str(time_spent))
    return final

def emulate_arbiters(spikes, arbi_type):
    #Set arbiter logic and priorities here
    global acks, img_size
    if arbitrate:
        arbi = arbiter(spikes, delta_thres)
        if arbi_type=="FIFO":
            spikes=arbi.arbi_fixed()
        elif arbi_type=="RAND":
            spikes=arbi.arbi_rand()
        elif arbi_type=="FAIR":
            spikes=arbi.arbi_fair()
    else:
        acks=np.ones(img_size[::-1], dtype='bool') #set acks to all true if no arbitration takes place
    return spikes

def displayimg(name, data, winsize, move):
    size=img_size[0]*winsize
    cv2.imshow(name, data)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size, size)
    cv2.moveWindow(name, move[0], move[1])

def img():
    global img_size, final_array, ref_array, priorities, acks
    ogs = time.time()
    factor=2 #used for increasing opencv window sizes
    imgr = check_img(filename)
    img_size = tuple([int(i / downscale_factor) for i in imgr.shape])
    img = cv2.resize(imgr, img_size, interpolation=cv2.INTER_AREA)
    final_array = np.zeros(img_size[::-1], dtype="uint8")  # constructing final image/s
    ref_array = np.zeros(img_size[::-1], dtype="uint8")  # reference array, previous frame/old value
    #Cumulative priority measurement
    priorities=np.zeros(img_size[::-1], dtype="double")
    acks=np.zeros(img_size[::-1], dtype="bool")
    displayimg('initial', img, factor, (500, 400))
    print("file & img check takes: " + str(time.time() - ogs) + "sec")
    final = frame(img)
    ogf = time.time()
    print("Actual exec time= " + str(ogf - ogs) + "sec")
    displayimg('output', final, factor, (628, 400))
    cv2.waitKey()
    cv2.destroyAllWindows()

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
    global img_size, final_array, ref_array, priorities, acks
    #ogs = time.time()
    factor=2 #used for increasing opencv window sizes
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
    acks = np.zeros(img_size[::-1], dtype="bool")
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
    #global final_array, ref_array, img_size
    if not video:
        if is_file(filename):
            img()
    else:
        #set up webcam/video readout
        if is_file(filename):
            video_out()


if __name__ == "__main__":
    main()

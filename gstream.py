import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import torch
import numpy as np
import cv2
from torchvision import transforms
from main import CNNModel

frame = None
frame_list = []
frames_processed = 0

def on_frame_probe(pad, info):
    buf = info.get_buffer()
    global frame
    frame = buffer_to_tensor(buf, pad.get_current_caps())
    return Gst.PadProbeReturn.OK

def buffer_to_tensor(buf, caps):
    caps_structure = caps.get_structure(0)
    height, width = caps_structure.get_value('height'), caps_structure.get_value('width')
    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
            try:  
                image_array = np.ndarray(
                    (height, width, 4),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                return image_array[:,:,:3].copy()
                
            finally:
                buf.unmap(map_info)

Gst.init(None)

frame_format = 'BGRA'


'''

'''
gst_str = f'''
    v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw, format={frame_format}, framerate=(fraction)30/1 ! appsink name=s
'''

pipeline = Gst.parse_launch(gst_str)

pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

labels_list = ['all', 'baby', 'call', 'dead', 'eat', 'face', 'game', 'hello', 'idea', 'join', 'kiss', 'like', 'many',
               'name', 'orange', 'play', 'quiet', 'right', 'study', 'table', 'ugly', 'visit', 'walk', 'you', 'zero']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load('checkpoint.pth'))
model = model.eval()
text = None
color = None
word = None
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        rsframe = cv2.resize(frame, (112,112))
        frame_list.append(rsframe)
        if len(frame_list) == 25:
            frames = torch.stack([transforms.functional.to_tensor(frame) for frame in frame_list])
            frames = frames.permute(1,0,2,3)
            frames = frames.to(device)
            output = model(frames.unsqueeze(0))
            _, pred = torch.max(output.data, 1)
            index = int(pred)
            word = labels_list[index]
            frame_list = []
        cv2.putText(frame, word, (100,100), font, 4, color, 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break    
pipeline.set_state(Gst.State.NULL) 
cv2.destroyAllWindows()
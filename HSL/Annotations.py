import os
from PIL import Image, ImageFont, ImageDraw
from .Visualization import visualization
import cv2
import natsort
from .Utilities import frames_to_video



def create_handshape_overlay(image_to_annotate,handshape, root, frame_dir):
    output_directory = root
       
    #font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 80)
    font = ImageFont.load_default()
    img = Image.open(image_to_annotate)
    draw = ImageDraw.Draw(img)
    overlay = handshape
    position = (40,100)
    draw.text(position, str(overlay), (255,255,0), font=font)
    draw = ImageDraw.Draw(img)
    img.save(output_directory+str(image_to_annotate))                   
    #os.remove('./Outputs/openpos/result_'+str(j)+'.png')
    
    

    
class annotate(object):
    """
    Object to create video with handshape classification overlay
    """
    def __init__(self, json_directory, frame_directory,  *args, **kwargs):
        """    
        parameters:
        -----------
        json_directory (ex. /data/),
        frame_directory
        optional:
        elbow: True/False,
        range_n_clusters = [] | default = 15
        """
        self._directory = json_directory
        self._frame_directory = frame_directory
        self._range_n_clusters = kwargs.get('range_n_clusters', 15)
        
        
    @property
    def video(self):
        frames = visualization(self._directory, self._frame_directory)._sorted_frame_files
        kmeans = visualization(self._directory, self._frame_directory, range_n_clusters = self._range_n_clusters).run_k_means
        
        output_directory = "./Output_video/"
        
        if not (os.path.isdir(output_directory)):
            os.mkdir(output_directory)
            os.mkdir(output_directory+self._frame_directory)
            print("Output directory created..")
        
        for i in range(len(frames)-1):
            create_handshape_overlay(self._frame_directory+frames[i],kmeans[0][i], output_directory, self._frame_directory)
        print("Frames with cluster overlay created..")
        
        print("Creating output video..")
        frames_to_video(output_directory+self._frame_directory, 'jpg', './output_video.mp4')
    
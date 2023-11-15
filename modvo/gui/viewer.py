import cv2
import pangolin
import OpenGL.GL as gl
import numpy as np
from multiprocessing import Process, Queue, Value

class MapElement():
    def __init__(self):
        self.points = None

class TrajectoryElement():
    def __init__(self):
        self.poses = None
        self.images = None

class GUIDrawer():
    """
    This class is responsible for drawing the GUI.   
    """
    def __init__(self):
        self.window_size = (1024, 550)
        self.trajectory_queue = Queue()
        self.map_queue = Queue()
        self.current_traj = None
        self.current_map = None
        self.drawer_process = Process(target=self.viewer_thread)
        self.drawer_process.daemon = True
        self.drawer_process.start()
        
        
    def start(self):
        """
            Initialize the GUI
        """
        width =  self.window_size[0]
        height =  self.window_size[0]
        pangolin.CreateWindowAndBind('VO/VSLAM Viewer', width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        viewpoint_x = 0
        viewpoint_y = -40
        viewpoint_z = -80
        viewpoint_f = 1000
        # Define Projection and initial ModelView matrix
        proj = pangolin.ProjectionMatrix(width, height, viewpoint_f, viewpoint_f, width//2, height//2, 0.1, 5000)
        look_at = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam = pangolin.OpenGlRenderState(proj, look_at)
        
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -width/height)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))
        
        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()
        

    def update(self):

        while(not self.trajectory_queue.empty()):
            self.current_traj = self.trajectory_queue.get()
        self.scam.Follow(self.Twc, True)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dcam.Activate(self.scam)

        if(self.current_traj is not None):
            if(len(self.current_traj.poses) > 1):
                #draw the cameras in black
                gl.glColor3f(0.0, 0.0, 0.0)
                pangolin.DrawCameras(self.current_traj.poses[:-1])

            #current camera in green
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawCameras(self.current_traj.poses[-1:])    
            self.Twc.m = self.current_traj.poses[-1]
            
            #show current image
            disp_img = cv2.resize(self.current_traj.image, (640, 480))
            cv2.imshow('Mod-VO: current image', disp_img)
            cv2.waitKey(1)

        while(not self.map_queue.empty()):
            self.current_map = self.map_queue.get()

        if(self.current_map is not None and len(self.current_map.points) > 0):
            gl.glPointSize(5.0)
            gl.glColor3f(0.0, 0.0, 0.0)
            pangolin.DrawPoints(self.current_map.points)

        pangolin.FinishFrame()


    def quit(self):
        self.drawer_process.join()
    

    def viewer_thread(self):
        self.start()
        while(not pangolin.ShouldQuit()):
            self.update()

        print('Closing GUI...')
    
    
    def draw_trajectory(self, frames):
        if self.trajectory_queue is None:
            return
        trajectory = TrajectoryElement()
        trajectory.poses = [np.array(f.pose) for f in frames]
        trajectory.image = frames[-1].image.copy()
        self.trajectory_queue.put(trajectory)
    
    def draw_map_points(self, points):
        if self.map_queue is None:
            return
        map_obj = MapElement()

        map_obj.points = np.array([p.coordinates for p in points])
        self.map_queue.put(map_obj)
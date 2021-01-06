import cv2
import numpy as np
# import matplotlib.pyplot as plt


def preprocess(img):
    h,w = img.shape[:2]
    # log function help pixel low contrast lighting situations
    img = np.log(np.float32(img) + 1.0)
    # normalize pixel value  
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # used hanning window 
    win_col = np.hanning(w)
    win_row = np.hanning(h)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win_hanning = mask_col * mask_row
    img = img * win_hanning

    return img
    
def linear_mapping(img):
    return (img - np.min(img)) / (np.max(img)-np.min(img))

def random_warp(img):
    """
    Random rotate object
    """
    h,w = img.shape[:2]
    T = np.zeros((2,3))
    coef = 0.2
    ang = (np.random.rand()-0.5) * coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2,:2] = [[c,-s],[s,c]]
    T[:2,:2] += (np.random.rand(2,2) - 0.5)* coef
    c = (h//2, w//2)
    T[:,2] = c - np.dot(T[:2,:2], c)

    return cv2.warpAffine(img, T, (w,h), borderMode= cv2.BORDER_REFLECT)

def drawbox(img, box):
    x,y,h,w = [int(i) for i in box]
    cv2.putText(img, 'OpenCV MOSSE' , (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w,y+h), (255,0,0), 1)

class MosseTracking():
    def __init__(self, cam, eta= 0.125, rotate=True, num_pretrain= 128):
        self.eta = eta
        self.window_name = 'Object Tracking'
        self.rotate = rotate
        self.num_pretrain = num_pretrain
        self.cam = cam
        _, frame = self.cam.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.pos_obj = cv2.selectROI(self.window_name, frame, False)
        self.pos_obj = np.array(self.pos_obj).astype(np.int64)
        x,y,w,h = [int(i) for i in self.pos_obj]
        self.fi = frame_gray[y:y+h, x:x+w]
        self.gt = frame_gray[y:y+h, x:x+w]

        self.g = self.generate_gauss_gt(self.gt)
        self.G = np.fft.fft2(self.g)
        self.Ai, self.Bi = self.pre_training(self.fi, self.G)

    def tracking(self):
        """ 
        """
        count = 0
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break
            # first frame
            if count == 0:
                Ai = self.eta * self.Ai
                Bi = self.eta * self.Bi
                pos = self.pos_obj.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:                     
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = preprocess(cv2.resize(fi, (self.pos_obj[2], self.pos_obj[3])))
                Gi = Hi * np.fft.fft2(fi)
                # gi = np.fft.ifft2(Gi)
                gi = linear_mapping(np.fft.ifft2(Gi))
                # find the max pos
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                # update the position
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin, ymin, xmax, ymax]
                clip_pos[0] = np.clip(pos[0], 0, frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)
                # get the current fi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = preprocess(cv2.resize(fi, (self.pos_obj[2], self.pos_obj[3])))
                # update Ai, Bi
                Ai = self.eta * (self.G * np.conjugate(np.fft.fft2(fi))) + (1 - self.eta) * Ai
                Bi = self.eta * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.eta) * Bi
            count += 1
            drawbox(frame, pos)
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    def pre_training(self, obj, G):
        """
        """
        h,w = G.shape[:2]
        fi = cv2.resize(obj, (h,w))
        Ai = np.zeros_like(G)
        Bi = np.zeros_like(G)
        for i in range(self.num_pretrain):
            if self.rotate:
                fi = preprocess(random_warp(obj))
            else:
                fi = preprocess(obj)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) 
        return Ai, Bi

    def generate_gauss_gt(self, gt):
        """ 
        """
        h,w = self.gt.shape[:2]
        g = np.zeros((h, w))
        g[h//2, w//2] =1
        g = cv2.GaussianBlur(g, (-1,-1), 2.0)
        g /= g.max()
        
        return g

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)

    tracker = MosseTracking(cam)
    tracker.tracking()

#Ftech AI
#
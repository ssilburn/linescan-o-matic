
# Get the directory this python fils is in.
import os
current_path = os.path.dirname(os.path.realpath(__file__))

# GUI modules
from PyQt4 import QtCore, QtGui, uic

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as MPLToolbar
import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

# Matplotlib canvas widget (from matplotlib website example)
class MplCanvas(FigureCanvas):
   

    def __init__(self, parent=None, width=5, height=4, dpi=75):

        self.fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


# Class for the window
class InterfaceWindow(QtGui.QMainWindow):
 
    def __init__(self, app, parent = None):

        # GUI initialisation
        QtGui.QDialog.__init__(self, parent)
        # Load the Qt designer file, assumed to be in the same directory as this python file and named gui.ui.
        uic.loadUi(os.path.join(current_path,'main_window.ui'), self)

        self.mpl_canvas = MplCanvas(self.centralwidget)
        self.mpltoolbar = MPLToolbar(self.mpl_canvas,self)
        self.centralwidget.layout().insertWidget(0,self.mpl_canvas)
        self.centralwidget.layout().insertWidget(0,self.mpltoolbar)
        self.aspect_slider.valueChanged.connect(self.change_scanspeed)
        self.manual_speed_adjust.toggled.connect(self.change_scanspeed)
        self.max_interp.valueChanged.connect(self.change_scanspeed)
        self.app = app

        # Callbacks for GUI elements: connect the buttons to the functions we want to run
        self.action_load_vid.triggered.connect(self.open_vid)
        self.action_save.triggered.connect(self.save_image)

        self.speed_detect_shift = 5
        self.speed_detect_window = 20
        self.auto_xmap = None

        # Start the GUI!
        self.show()


    def open_vid(self):

        filedialog = QtGui.QFileDialog(self)
        filedialog.setAcceptMode(0)
        filedialog.setFileMode(1)

        filedialog.setWindowTitle('Load Video...')
        filedialog.setNameFilter('Video Files (*.mp4 *.avi)')
        filedialog.exec_()
        if filedialog.result() == 1:
            vid_path = filedialog.selectedFiles()[0]
            vid_dialog = LoadVideoDialog(self,vid_path)
            vid_dialog.exec_()

            if vid_dialog.result() == 1:
                self.xmap = None
                self.manual_speed_adjust.blockSignals(True)
                self.manual_speed_adjust.setChecked(True)
                self.manual_speed_adjust.blockSignals(False)
                vid = cv2.VideoCapture(vid_path)
                n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) 
                h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if vid_dialog.select_col.isChecked():
                    self.image = np.zeros((h,n_frames,3),np.uint8)
                    self.im_plus = np.zeros((h,n_frames,3),np.uint8)
                    self.im_minus = np.zeros((h,n_frames,3),np.uint8)
                else:
                    self.image = np.zeros((w,n_frames,3),np.uint8)
                    self.im_plus = np.zeros((w,n_frames,3),np.uint8)
                    self.im_minus = np.zeros((w,n_frames,3),np.uint8)

                self.mpl_canvas.fig.clear()
                self.imax = self.mpl_canvas.fig.add_subplot(111)

                self.display_im = self.imax.imshow(self.image)
                self.imax.set_axis_off()

                self.output_dims.setText('{:d} x {:d} ({:.1f} Mpx)'.format(self.image.shape[1],self.image.shape[0],np.prod(self.image.shape)/1e6))

                time_ind = -1

                space_ind = vid_dialog.line_index.value()

                rtl = vid_dialog.rtl.isChecked()
                output_width = self.image.shape[1]
                self.app.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
                while time_ind < n_frames-1:
                    time_ind = time_ind + 1
                    if rtl:
                        im_col_ind = output_width - time_ind - 1
                    else:
                        im_col_ind = time_ind
                    self.statusbar.showMessage('Reading video frames: {:.0f}%...'.format(100.*time_ind/n_frames))
                    ret,frame = vid.read()
                    if frame is None:
                        continue
                    else:
                        frame[:,:,:3] = frame[:,:,2::-1]

                        if vid_dialog.select_col.isChecked():
                            self.image[:,im_col_ind,:] = frame[:,space_ind,:]
                            self.im_plus[:,im_col_ind,:] = frame[:,space_ind+self.speed_detect_shift,:]
                            self.im_minus[:,im_col_ind,:] = frame[:,space_ind-self.speed_detect_shift,:]
                        elif vid_dialog.select_row.isChecked():
                            self.image[:,im_col_ind,:] = frame[space_ind,:,:]
                            self.im_plus[:,im_col_ind,:] = frame[space_ind+self.speed_detect_shift,:,:]
                            self.im_minus[:,im_col_ind,:] = frame[space_ind-self.speed_detect_shift,:,:]

                    if not time_ind % 10:
                        self.display_im.set_data(self.image)
                        self.mpl_canvas.draw()
                        self.app.processEvents()

                vid.release()
                self.im_out = self.image
                self.app.restoreOverrideCursor()
                self.statusbar.clearMessage()


    def change_scanspeed(self):

        if self.manual_speed_adjust.isChecked():
            self.aspect_slider.setEnabled(True)
            aspect = 1 + 0.02*self.aspect_slider.value()
            xmap = np.linspace(0,self.image.shape[1]-1,int(self.image.shape[1]/aspect))

        elif self.auto_speed_adjust.isChecked():
            self.aspect_slider.setEnabled(False)
            if self.auto_xmap is None:
                self.app.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
                speed = []
                refpos = []
                for i in range(0,self.image.shape[1]-self.speed_detect_window,self.speed_detect_window):
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(self.im_minus[:,i:i+self.speed_detect_window],self.im_plus,cv2.TM_CCORR_NORMED))
                    if abs(max_loc[0] - i) > 0: 
                        speed.append( abs( 2*self.speed_detect_shift / (max_loc[0] - i) ) )
                    else:
                        speed.append(0)
                    refpos.append(i)
                    self.app.processEvents()
                    self.statusbar.showMessage('Detecting scan speed: {:.0f}%...'.format(100*i/self.image.shape[1]))
                self.app.restoreOverrideCursor()
                refpos = np.array(refpos)
                speed = np.array(speed)                

                fill_speed = 1.

                speed[speed < 0.1] = fill_speed

                newx = cumtrapz(speed,x=refpos,initial=0)
                self.auto_xmap = interp1d(newx,refpos)(np.arange(0,newx.max()))
                self.statusbar.clearMessage()

            xmap = self.auto_xmap
        

        factor = min(1.,np.gradient(xmap).min() * self.max_interp.value()) 
        xmap = interp1d(np.arange(xmap.size),xmap,kind='cubic')(np.linspace(0,xmap.size-1,int(xmap.size*factor)))

        ymap = np.tile( np.linspace(0,self.image.shape[0]-1,int(self.image.shape[0]*factor))[:,np.newaxis],[1,xmap.size] ).astype('float32')
        xmap = np.tile(xmap[np.newaxis,:],[ymap.shape[0],1]).astype('float32')

        self.im_out = cv2.remap(self.image,xmap,ymap,cv2.INTER_CUBIC)

        self.output_dims.setText('{:d} x {:d} ({:.1f} Mpx)'.format(self.im_out.shape[1],self.im_out.shape[0],np.prod(self.im_out.shape)/1e6))

        self.mpl_canvas.fig.clear()
        self.imax = self.mpl_canvas.fig.add_subplot(111)

        self.display_im = self.imax.imshow(self.im_out)
        self.imax.set_axis_off()
        self.mpl_canvas.draw()



    def save_image(self):

        filedialog = QtGui.QFileDialog(self)
        filedialog.setAcceptMode(1)
        
        filedialog.setFileMode(0)

        filedialog.setWindowTitle('Save As...')
        filedialog.setNameFilter('JPEG Image (*.jpg)')
        filedialog.exec_()

        if filedialog.result() == 1:
            fname = filedialog.selectedFiles()[0]
            im = self.im_out.copy()
            im[:,:,:3] = im[:,:,2::-1]

            if not fname.endswith('.jpg'):
                fname = fname + '.jpg'

            cv2.imwrite(fname,im)




class LoadVideoDialog(QtGui.QDialog):

    def __init__(self, parent,vid_fname):

        # GUI initialisation
        QtGui.QDialog.__init__(self, parent)
        uic.loadUi(os.path.join(current_path,'load_video.ui'), self)

        self.mpl_canvas = MplCanvas(self.im_frame)
        self.im_frame.layout().addWidget(self.mpl_canvas)

        vid = cv2.VideoCapture(vid_fname)
        self.w = vid.get(cv2.CAP_PROP_FRAME_WIDTH) 
        self.h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.nframes.setText('{:.0f}'.format(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fdims.setText('{:.0f} x {:.0f} px'.format(self.w,self.h))

        if self.select_col.isChecked():
            self.line_index.setMinimum(5)
            self.line_index.setMaximum(self.w-6)
            self.line_index.setValue(self.w/2)
        elif self.select_row.isChecked():
            self.line_index.setMinimum(5)
            self.line_index.setMaximum(self.h-6)
            self.line_index.setValue(self.h/2)

        self.line_index.valueChanged.connect(self.update_line)

        self.select_col.toggled.connect(self.update_line)

        exframe = None
        while exframe is None:
            ret,exframe = vid.read()

        exframe[:,:,:3] = exframe[:,:,2::-1]

        self.imax = self.mpl_canvas.fig.add_subplot(111)
        self.mpl_canvas.fig.subplots_adjust(0,0,1,1)
        self.imax.imshow(exframe)
        self.imax.set_axis_off()

        self.line_disp = self.imax.plot([self.line_index.value(),self.line_index.value()],[0,self.h-1],'r')[0]

        vid.release()

        self.mpl_canvas.mpl_connect('button_release_event',self.move_line)


    def move_line(self,event):
        if self.select_col.isChecked():
            self.line_index.setValue(event.xdata)
        elif self.select_row.isChecked():
            self.line_index.setValue(event.ydata)

    def update_line(self):

        if self.select_col.isChecked():
            self.line_index.setMinimum(5)
            self.line_index.setMaximum(self.w-6)
            self.line_disp.set_data([self.line_index.value(),self.line_index.value()],[0,self.h-1])
        elif self.select_row.isChecked():
            self.line_index.setMinimum(5)
            self.line_index.setMaximum(self.h-6)
            self.line_disp.set_data([0,self.w-1],[self.line_index.value(),self.line_index.value()])

        self.mpl_canvas.draw()


if __name__ == '__main__':

	# Create a GUI application
    app = QtGui.QApplication([])
    
    # Create an instance of InterfaceWindow
    window = InterfaceWindow(app)
    
    # Run the interface
    app.exec_()

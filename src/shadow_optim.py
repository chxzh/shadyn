import glfw
import numpy as np
from OpenGL.GL import *
import tools
from app import *
from ctypes import c_uint8, c_float, c_ushort, c_void_p
from math import pi, cos, sin
from cgkit.cgtypes import *
from PIL import Image
import cma
from scipy import optimize
from astropy.units import count


from PySide.QtGui import *
from PySide.QtCore import *
from threading import Thread, Lock
import sys
import random
from astropy.convolution.boundary_extend import DTYPE
import log, plotting
from functools import wraps
from tools import get_fname
# log.init()

qt_app = QApplication(sys.argv)
class MyGUI(QWidget):
    def __init__(self, renderer):
        QWidget.__init__(self)
        self.renderer = renderer        
        self.renderer.wait_till_init()
        _init_X_Y(renderer.window.width, renderer.window.height)
        self._init_window()
        vbox = QVBoxLayout()
        left_boxes = [
                 self._init_set_target(),
                 self._init_select_method(),
                 self._init_select_energy(),
                 self._init_set_param()
                 ]
        for box in left_boxes:
            vbox.addLayout(box)
        vbox.addStretch(1)
        vbox.addLayout(self._init_main_control())
        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        
        vbox = QVBoxLayout()
        right_boxes = [
                       self._init_target_preview()
                       ]
        for box in right_boxes:
            vbox.addLayout(box)
        hbox.addLayout(vbox)
        self.setLayout(hbox)
        self._optimizer = None
        self.target_path = None
        self._after_renderer_ready()
        
    def _after_renderer_ready(self):
        self._on_param_updated()

    def _init_window(self):
        self.move(0, 525)
        self.resize(500, 500)
        self.setWindowTitle("Shadow Optimization Console")

    def _init_set_target(self):
        hbox = QHBoxLayout()
        self.load_target_button = QPushButton("load from file", self)
        QObject.connect(self.load_target_button, SIGNAL("clicked()"), self._on_set_target_file)
        hbox.addWidget(self.load_target_button)
        self.target_path_label = QLabel("target not set", self)
        self.target_path_label.setWordWrap(True)
        hbox.addWidget(self.target_path_label)
        hbox.addStretch(1)
        gbox = QGroupBox("target shadow", self)
        gbox.setLayout(hbox)
        temp_box = QHBoxLayout()
        temp_box.addWidget(gbox)
        return temp_box
    
    def _on_set_target_file(self):
        filename, filter = QFileDialog.getOpenFileName(self, 
                                        "select a target image", "",
                                        "image file (*.png *.jpg *.bmp)")
        self.target_path = filename
        self.target_path_label.setText(filename)
        self.target_preview_label.setPixmap(QPixmap(filename))
        self.target_preview_label.show()
        # TODO: deal with cases when the image is unfit
    
    def _init_target_preview(self):
        self.target_preview_label = QLabel("this is where preview of the target image should be showed",self)        
        self.target_preview_label.setGeometry(10, 30, 640, 480)
        vbox = QVBoxLayout()
        vbox.addWidget(self.target_preview_label)
        gbox = QGroupBox("target preview", self)
        gbox.setLayout(vbox)
        temp_box = QVBoxLayout()
        temp_box.addWidget(gbox)
        return temp_box
            
    def _init_main_control(self):
        self.play_pause_button = QPushButton("PLAY", self)
        QObject.connect(self.play_pause_button, SIGNAL("clicked()"), self._on_play_pause)
        self.is_on_going = False              
        self.stop_button = QPushButton("STOP", self)
        self.stop_button.setEnabled(False)
        QObject.connect(self.stop_button, SIGNAL("clicked()"), self._on_stop)
        hbox = QHBoxLayout()
        hbox.addWidget(self.play_pause_button)
        hbox.addWidget(self.stop_button)
        hbox.addStretch(1)
        return hbox

    def _on_play_pause(self):
        # not started
        if self._optimizer == None or not self._optimizer.is_alive():
            self._optimizer = Optimizer(self.renderer)
            plotter = plotting.Plotter(*get_fname("..\\res"))
            plotting.attach_plotter(self._optimizer, plotter)
            # configuring the optimizer by feeding in optimizing-method and energy function
            if self.target_path != None:
                self._optimizer.set_target(self.target_path)
            else:                
                msg_box = QMessageBox()
                msg_box.setText("no target image is selected")
                msg_box.exec_()
                return
            self._optimizer.set_method(self.optim_combo.currentText())
            try:
                self._optimizer.set_energy(*self._get_energy_pairs())
            except RuntimeWarning as rtwng:
                msg_box = QMessageBox()
                msg_box.setText("no energy function is selected")
                msg_box.exec_()
                return
            self._optimizer.set_finished_callback(self._on_optim_done)
            param_lock = Lock()
            self._optimizer.set_iter_callback(self._on_iter_callback, param_lock)
            param_updater = Lock_listener(param_lock, self._on_param_updated)
            if self.armijo_check.isEnabled():
                self._optimizer.line_search_first = self.armijo_check.checkState()
            self._optimizer.start()            
            self.play_pause_button.setText("PAUSE")
            self.stop_button.setEnabled(True)
            self.is_on_going = True
        else:
            # started already
            self.is_on_going = not self.is_on_going
            self._optimizer.switch()
            if self.is_on_going: # optimizing, to pause
                self.play_pause_button.setText("PLAY")
            else: # pausing, to continue
                self.play_pause_button.setText("PAUSE")
    
    def _on_stop(self):
        # TODO: find a way to stop the optimization
        if self.is_on_going: # the optimization is not paused
            # locking the green light to pause the current optimization
            self._optimizer.green_light.acquire()
        self._optimizer = None
        self._on_optim_done()
        
    def _on_optim_done(self):
        # this method is called when an optimization is done        
        self.play_pause_button.setText("PLAY")
        self.stop_button.setEnabled(False)
        self.is_on_going = False
        pass
    
    def _on_iter_callback(self, param_lock):
        # this method is a callback method for optimization thread, it
        # would be called at each round of iteration, by the end of
        if param_lock.locked(): # for the first round
            param_lock.release()
        pass

    def _init_select_method(self):
        self.optim_gbox = QGroupBox("optimization method",self)
        self.optim_combo = QComboBox(self)
        items = Optimizer.method_dic.keys()
        items.sort()
        self.optim_combo.addItems(items)
        self.optim_combo.currentIndexChanged.connect(self._on_method_index_changed)
        self.armijo_check = QCheckBox(self)
        self.armijo_check.setText("apply Armijo's Rule on line search")
        if not self.optim_combo.currentText() in Optimizer.linear_search_list:
            self.armijo_check.setEnabled(False)
        vbox = QHBoxLayout()
        vbox.addWidget(self.optim_combo)
        vbox.addWidget(self.armijo_check)
        vbox.addStretch(1)
        self.optim_gbox.setLayout(vbox)
        temp_box = QVBoxLayout()
        temp_box.addWidget(self.optim_gbox)
        return temp_box
    
    def _on_method_index_changed(self, index):
        method_name = self.optim_combo.itemText(index)
        if method_name in Optimizer.linear_search_list:
            self.armijo_check.setEnabled(True)
        else:
            self.armijo_check.setEnabled(False)
    
    def _init_select_energy(self):
        self.energy_gbox = QGroupBox("energy functions", self)
        vbox = QVBoxLayout()
        energy_names = Optimizer.energy_dic.keys()
        energy_names.sort()
        self.energy_chbox_list = []    
        self.weights = []    
        for index, energy_name in enumerate(energy_names):
            checkbox = QCheckBox(self)
            checkbox.setText(energy_name)
            vbox.addWidget(checkbox)
            self.energy_chbox_list.append(checkbox)
            self.weights.append(0)
            checkbox.stateChanged.connect(
                    self._on_energy_state_changed_closure(index, checkbox))
        self.checked_num = 0
        self.weights = np.array(self.weights, dtype=float)
        hbox = QHBoxLayout()
        self.weight_button = QPushButton("set weights", self)
        self.weight_button.clicked.connect(self._on_weight_button_clicked)
        hbox.addWidget(self.weight_button)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        self.energy_gbox.setLayout(vbox)
        temp_box = QVBoxLayout()
        temp_box.addWidget(self.energy_gbox)
        return temp_box
    
    def _on_energy_state_changed_closure(self, index, checkbox):
        def _on_state_changed(arg__1):
            # unaffected items should maintain the original balance
            # all weights sum-up to 1 except all-0
            if checkbox.checkState():
                # now checked, unchecked previously
                self.checked_num += 1
                self.weights *= (self.checked_num-1)
                self.weights[index] = 1.0
                self.weights /= self.checked_num
            else:
                # now unchecked, checked previously
                self.checked_num -= 1                
                self.weights *= (self.checked_num+1)
                self.weights[index] = 0.
                if self.checked_num != 0:
                    self.weights /= self.checked_num
            return
        return _on_state_changed
    
    def _on_weight_button_clicked(self):
        if False: # unselected, prompt to ask
            pass
        else:
            err_func_names = []
            cur_weights = []
            for checkbox, weight in zip(self.energy_chbox_list, self.weights):
                if checkbox.isChecked():
                    err_func_names.append(checkbox.text())
                    curweight.append(weight)
            names, weights = Weight_Dialog.get_weights(self, err_func_names, cur_weights)
            # TODO: finish this
        
        
    def _get_energy_pairs(self):
        # return the energy function selection and corresponding weights
        names, weights = [], []
        for checkbox, weight in zip(self.energy_chbox_list, self.weights):
            if checkbox.isChecked():
                names.append(checkbox.text())
                weights.append(weight)
        if len(names) == 0: raise RuntimeWarning("No energy function selected")
        return names, weights
    
    def _on_param_updated(self):
        # update the fields and bars
        params = self.renderer.get_param()
        self._update_fields(params)
            
    def _update_fields(self, params):
        for param, field_bundles in zip(params, self.param_fields):
            edit, slider, param2val = field_bundles 
            edit.setText(str(param))
            slider.setValue(param2val(param))

    def _init_set_param(self):
        vbox = QVBoxLayout()
        # attribute := (name, slider_min, slider_max)
        example_x = self.renderer.get_param()
#         self.param_attributes = [("cube x-coord", -5, 5), 
#                                  ("cube y-coord", -5, 5), 
#                                  ("cube z-coord", -5, 5)]
        self.param_attributes = [(("param %d"%i),-5,5) for i, x_i in enumerate(example_x)]
        self.param_fields = []
        for index, param_attribute in enumerate(self.param_attributes):
            name, slider_min, slider_max = param_attribute
            param_label = QLabel(name, self)
            param_field = QLineEdit(self)
            param_field.setValidator(QDoubleValidator(parent=param_field))
            param_slider = QSlider(Qt.Horizontal, self)
            param_slider.setTickPosition(QSlider.TicksBelow)
            hbox = QHBoxLayout()
            hbox.addWidget(param_label)
            hbox.addWidget(param_field)
            hbox.addWidget(param_slider)
            hbox.setStretch(2,1)
            vbox.addLayout(hbox)
            _param2val, _val2param = self._get_param_exchanger(slider_min, slider_max)            
            param_slider.valueChanged.connect(
                    self._on_slider_value_changed_closure(index, param_field, _param2val, _val2param))
            param_field.returnPressed.connect(
                    self._on_edit_returned_closure(index, param_field, param_slider, _param2val, _val2param))
            self.param_fields.append((param_field, param_slider, 
                                      _param2val))
        hbox = QHBoxLayout()
        self.rand_param_button = QPushButton("randomize", self)
        QObject.connect(self.rand_param_button, SIGNAL("clicked()"),
                        self._on_randomize)        
        hbox.addWidget(self.rand_param_button)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        gbox = QGroupBox("parameters")
        gbox.setLayout(vbox)
        temp_box = QVBoxLayout()
        temp_box.addWidget(gbox)
        return temp_box
    
    def _on_slider_value_changed_closure(self, index, field, param2val, val2param):
        def _on_value_change(value):
            cur_param = float(field.text())
            if param2val(cur_param) == value:
                # slider unchanged, shouldn't mess with line edit
                return
            param = val2param(value)
            field.setText(str(param))
            self.renderer.set_param_indiv(param, index)
        return _on_value_change
    
    def _on_edit_returned_closure(self, index, field, slider, param2val, val2param):
        def _on_returned():
            param = float(field.text())
            slider.setValue(param2val(param))
            self.renderer.set_param_indiv(param, index)
        return _on_returned
    
    def _get_param_exchanger(self, param_min, param_max):
        param_min, param_max = min(param_min, param_max), max(param_min, param_max)
        exchange_rate = 99.0 / (param_max - param_min)
        def _get_value(param):
            return max(0, min(int((param-param_min)*exchange_rate),99))
        def _get_param(value):
            # value ranges within [0, 99]
            return float(param_min + value/exchange_rate)
        return _get_value, _get_param
    
    def _on_randomize(self):
        new_param = [random.uniform(low, high) for name, low, high in self.param_attributes]
        new_param = np.array(new_param)
        self.renderer.set_param(new_param)
        self._update_fields(new_param)
        pass
        
    def run(self):
        self.show()
        qt_app.exec_()
    
    ### End of MyGUI ###


class Lock_listener(Thread):
    def __init__(self, lock, callback):
        Thread.__init__(self)
        self._lock = lock
        self._callback = callback
        self._terminate_flag = False
        self.start()
    
    def run(self):
        while True:
            self._lock.acquire()
            if self._terminate_flag: break
            self._callback()
        
    def terminate(self):
        self._terminate_flag = True
    
    ### End of Lock_listener ###       


class Weight_Dialog(QDialog):
    def __init__(self, names, weights):
        QDialog.__init__(self)
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        self.weight_slider = weight_slider(names, weights)
        hbox.addWidget(self.weight_slider)
        vbox.addLayout(hbox)
        self.confirm_button = QPushButton('Confirm')
        self.cancel_button = QPushButton('Cancel')
        
        btn_box = QHBoxLayout()
        btn_box.addStretch(1)
        btn_box.addWidget(self.confirm_button)
        btn_box.addWidget(self.cancel_button)
        vbox.addLayout(btn_box)
        self.setLayout(vbox)
    
    def paintEvent(self, event):
        pass
#         qp = QPainter()
#         qp.begin(self.confirm_button)
#         qp.setPen(QColor(168, 34, 3))
#         qp.setFont(QFont('Decorative', 10))
#         qp.drawText(event.rect(), Qt.AlignCenter, "fdafdsa")
#         qp.end()
         
    def run(self):
        self.exec_()
        
    @staticmethod
    def get_weights(parent=None, err_func_names=None, cur_weights=None):
        dialog = Weight_Dialog()
        dialog.exec_()
        return 
    ### End of Weight_Dialog ###
    
class weight_slider(QWidget):
    def __init__(self, names, weights):
        QWidget.__init__(self)
        self.setMinimumHeight(50)
        self.setMaximumHeight(100)
        self.setFixedWidth(400)
        self.names = names
        self.weights = weights
        self.bar_height = 50
        n = len(names)
        # chunk := (name, weight, color, (start, end))
        self.chunks = zip(names, weights, 
                          self.color_generator(n), self.chunk_generator(weights))
    
    def color_generator(self, n):
        yield QColor.fromHsv(0, 255, 255)
        i = 1
        bar = 2
        while i < n:
            hue = 360 * ((i - bar / 2) * 2 + 1) / bar
            color = QColor.fromHsv(hue, 255, 255)
            i += 1
            if i >= bar: bar *= 2
            yield color
    
    def chunk_generator(self, weights):
        start = 0.0
        roundint = lambda x: int(round(x))
        for weight in weights:
            end = start + self.width()*weight
            yield (roundint(start), roundint(end)-1)
            start = end
    
    # overriding the widget paintEvent()
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.setBrush(QColor(168, 34, 3))
        qp.drawRect(self.rect())
        for name, weight, color, chunk in self.chunks:
            qp.setBrush(color)
            qp.setPen(color)
            start, end = chunk
            qp.drawRect(start, 0, end-start, self.bar_height-1)
            
#         qp.setPen(QColor(12,4,244))
#         qp.setBrush(QColor(168, 34, 3))
        
#         qp.drawRect(self.rect())
#         qp.setFont(QFont('Decorative', 10))
#         qp.drawText(event.rect(), Qt.AlignCenter, "fdafdsa")
        qp.end()
    
    def dragEnterEvent(self, *args, **kwargs):
        print "drag enter event emitted"
        return QWidget.dragEnterEvent(self, *args, **kwargs)
    
    def dragLeaveEvent(self, *args, **kwargs):
        print "drag leave event emitted"
        return QWidget.dragLeaveEvent(self, *args, **kwargs)
    
    def dragMoveEvent(self, *args, **kwargs):
        print "drag move event emitted"
        return QWidget.dragMoveEvent(self, *args, **kwargs)
    
    def mouseMoveEvent(self, *args, **kwargs):
#         print "mouse move event emitted"
        return QWidget.mouseMoveEvent(self, *args, **kwargs)
    
    def mousePressEvent(self, *args, **kwargs):
        print "mouse pressed"
        return QWidget.mousePressEvent(self, *args, **kwargs)

# obsolete, check out the one as an instance method of Optimizer
def cma_optimize(name): # name is unused
    sigma_0 = 0.1
    ftarget = 1e-4
    opts = cma.CMAOptions()
    opts['ftarget'] = ftarget
    def cma_fmin(f, x):
        res = cma.fmin(objective_function=f, x0=x, sigma0=sigma_0, options=opts)
        return res[0], res[1]
    return cma_fmin

def scipy_optimize(name):
    def fmin(f, x):
        res = optimize.minimize(fun=f, x0=x, method=name)
        return res.x, res.fun
    return fmin

def scipy_optimize_jac(name):
    delta = 0.005        
    def fmin(f, x):
        res = optimize.minimize(fun=f, x0=x, method=name,
                    jac=_get_jac(f, delta, x))
        return res.x, res.fun
    return fmin

def _get_jac(func, delta, x0):
    # a gradient approximated jacobian computation
    # let func be the energy function and delta as the uniform delta for gradient
    len_x = len(x0)
    def jac(x):
        fx = func(x)
        grad = np.zeros(len_x)        
        for i in range(len_x):
            x_t = np.zeros(len_x)
            x_t[i] = delta
            fx_t = func(x+x_t)
            grad[i] = fx_t - fx
        return grad / delta
    return jac

def _sq_diff(a, b):
    # calculate the square difference of two equal-shaped numpy array
    return ((a-b)**2).sum()

_X, _Y = None, None

def _init_X_Y(width, height):
    # will be called by renderer when the initialization is done
    global _X, _Y
    _X = np.arange(width).reshape(1, width)
    _Y = np.arange(height).reshape(height, 1)
        
def _get_sec_moments(image):
    # image should be a gray scale Image object
    img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
    # using 128 in case of gray
    img = img.astype(np.int8)
    width, height = image.size
    img = img.reshape(height, width)
    M_00 = float(img.sum())   
#     if _X == None or _Y == None:
#         _init_X_Y(width, height)     
    M_10 = (_X * img).sum()
    M_01 = (img * _Y).sum()
    m_10 = M_10 / M_00 if M_00 else 0
    m_01 = M_01 / M_00 if M_00 else 0
    X_offset = _X - m_10
    Y_offset = _Y - m_01
    m_20 = ((X_offset**2)*img).sum() / M_00 if M_00 else 0
    m_02 = (img*(Y_offset**2)).sum() / M_00 if M_00 else 0
    m_11 = (X_offset*img*Y_offset).sum() / M_00 if M_00 else 0
    return np.array([m_20, m_11, m_02])    
    
def _get_fst_moments(image):
    # image should be a gray scale Image object
    img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
    # using 128 in case of gray
    img = img.astype(np.int8)
    width, height = image.size
    img = img.reshape(height, width)
    M_00 = float(img.sum())        
#     if _X == None or _Y == None:
#         _init_X_Y(width, height)
    M_10 = (_X * img).sum()
    M_01 = (img * _Y).sum()
    m_10 = M_10 / M_00 / 640 if M_00 else 0
    m_01 = M_01 / M_00 / 480 if M_00 else 0
    return np.array([m_10, m_01])

from PIL import ImageMath as imath    
def _xor_closure(target):
    def _get_xor(image):
        xor_img = imath.eval("a^b", a=image, b=target)
        return sum(xor_img.getdata()) / (640*480)
    return _get_xor

def _sq_diff_closure(func):
    def sub_closure(target):
        res_t = func(target)
        def sqdiff(image):
            return _sq_diff(res_t, func(image))
        return sqdiff
    return sub_closure
    

class Optimizer(Thread):
    def __init__(self, renderer):
        Thread.__init__(self)
        # only when green light is unlocked will the optimization continue
        self.green_light = Lock()
        self.renderer = renderer
        self._optim_method = lambda *x: None
        self._energy_list = []
        self.set_param = renderer.set_param
        self.get_param = renderer.get_param
        self._target_img = None
        self._target_scores = {}
        self._finished_callback = lambda *args: None
        self._finished_callback_args = []
        self._iter_callback = lambda *args: None
        self._iter_callback_args = []
        self.line_search_first = False        
        self._method_name = "unset"
        self._eval_term_records = ()
        self._eval_sum_record = None
        self.result = None
    
    @log.task_log
    @plotting.plot_task
    def run(self):
        # check params
        if self._target_img == None:
            raise RuntimeError("Target image missing")
        if len(self._energy_list) == 0:
            raise RuntimeError("No energy function selected")
        if self._method_name == "unset":
            raise RuntimeError("Method unset")
        if self.renderer == None:
            raise RuntimeError("Renderer missing")
        # build optimizer
        err_func = self._wrap_eval(self.energy_func)
        if self.line_search_first:
            self.line_search_init_param(err_func)
        else:
            pass
        # wrap optimizer with green_light
        self.result = self._optim_method(x=self.renderer.get_param(),
                           f=err_func)
        self._finished_callback(*self._finished_callback_args)
        return self.result
        
    def line_search_init_param(self, func):
        x = self.renderer.get_param()
        jac = _get_jac(func=func, delta=0.005, x0=x)
        search_direction = - func(x)/jac(x)
        res = optimize.line_search(f=func, myfprime=jac, xk=x, pk=search_direction)
        alpha= res[0]
        x_new = x + alpha * search_direction
        self.renderer.set_param(x_new)
        
    def _wrap_eval(self, func):
        '''
        wrapping the evaluation function to be controllable by locks
        '''
        def wrapped(*x):
            self.green_light.acquire()
            self.green_light.release()
            return func(*x)
        return wrapped
    
    def switch(self):
        if self.green_light.locked():
            self.green_light.release()
        else:
            self.green_light.acquire()
    
    """
    method_dic is a mapping between optimizing method to a closure that takes
    the name (in some case like CMA, name doesn't affect anything) and returns
    the uniformed optimizing interface fmin(f, x0)
    """
    method_dic = {"CMA": cma_optimize,
                   "Nelder-Mead": scipy_optimize,
                   "Powell": scipy_optimize,
                   "CG": scipy_optimize_jac,
                   "BFGS": scipy_optimize_jac,
                   "Newton-CG": scipy_optimize_jac,
                   "L-BFGS-B": scipy_optimize_jac,
                   "TNC": scipy_optimize_jac,
                   "COBYLA": scipy_optimize,
                   "SLSQP": scipy_optimize_jac
#                    ,"dogleg": scipy_optimize_jac, # needs hessian, maybe later
#                    "trust-ncg": scipy_optimize_jac
                  }
    
    linear_search_list = ['CG', 'BFGS'] # TODO: finish the list
    
    def set_method(self, name):
        # this is the interface for manager
        if name == "CMA":
            self._optim_method = self._explict_CMA
        else:
            self._optim_method = Optimizer.method_dic[name](name)
        self._method_name = name
    
    def _explict_CMA(self, f, x):
        # it is very inappropriate to call them solutions as they are guesses
        # I'd rather call it samples or guesses but I yield
        def generate(solutions):
            for x in solutions:
                try:
                    yield f(x)
                except:
                    # like encountering an all-black error
                    self.renderer.stop()
                    self.renderer = Renderer()
                    self.renderer.start()
                    self.renderer.wait_till_init()
                    yield f(x)
        
        sigma_0 = 0.1
        ftarget = 2e-4
        opts = cma.CMAOptions()
        opts['ftarget'] = ftarget
        es = cma.CMAEvolutionStrategy(x, sigma_0, opts)
        while not es.stop():
            solutions = es.ask()
            fvals = [y for y in generate(solutions)]
            es.tell(solutions, fvals)
        res = es.result()
        return res[0], res[1]
    
    @plotting.init_plot
    def set_energy(self, func_names, weights):
        if len(func_names) != len(weights):
            err_msg = "energy function number (%d) doesn't match weight number (%d)"\
                    % (len(func_names), len(weights))
            raise RuntimeError(err_msg)
        
        self._energy_list = zip([Optimizer.energy_dic[name](self._target_img) \
                                        for name in func_names],
                                    func_names,
                                    weights)
        self._eval_term_records = tuple([] for i in func_names)
        self._eval_sum_record = []
            
    def set_target(self, image_path):
        # TODO: finish setting the target image
        self._target_img = Image.open(image_path)
        self._target_img = self._target_img.convert("L")
        # used to have other pre-computation when setting an target image
    
    def set_finished_callback(self, callback, *args):
        self._finished_callback = callback
        self._finished_callback_args = args
    
    def set_iter_callback(self, callback, *args):
        self._iter_callback = callback
        self._iter_callback_args = args
    
    penalty_weight = 1e-3
    def penalty(self, x):
        return sum(x**2) * self.penalty_weight
    
    def get_mat_model2snapshot(self, img):
        mat_shaj = self.renderer.shadow.shaject_mat
        mat_view = self.renderer.cam_cap.view_mat
        mat_proj = self.renderer.cam_cap.proj_mat
        w, h = img.size
        # mat_c2s is projection from clip coordinate to image coordinate
        mat_c2s = np.array([[(w-1)/2, 0, 0, (w-1)/2],
                            [0, (1-h)/2, 0, (h-1)/2],
                            [0,       0, 0,       0],
                            [0,       0, 0,       1]])
        mat_c2s *= np.array(mat_proj * mat_view * mat_shaj).T # cgkit.mat4 is column major
        return mat_c2s
    
    def model_center_penalty_closure(self, img):
        # presuming the camera for capturing, light and receiver won't move
        # or otherwise all the mat shall be computed on the fly
        mat_c2s = self.get_mat_model2snapshot(img)
        def model_center_penalty(x):
            positions = self._strip_positions(x)
            
            return 0
        return model_center_penalty

    def _strip_positions(self, x):
        return [tuple(x[0:3]), tuple(x[6:9]), tuple(x[12:15])]
    
    def _record_term(f):
        # decorator that records each term of each evaluation
        @wraps(f)
        def inner(cls, x):
            res = f(cls, x)
            for record, res_term in zip(cls._eval_term_records, res):
                y, w, n = res_term
                record.append(y)
            return res
        return inner
    
    def _record_sum(f):
        # decorator that records sum of all terms of each evaluation
        @wraps(f)
        def inner(cls, x):
            res = f(cls, x)
            cls._eval_sum_record.append(res)
            return res
        return inner
        
    def _sum(f):
        @log.energy_sum_log
        @wraps(f)
        def inner(cls, x):
            res = f(cls, x)
            return sum([w*y for y, w, n in res]) + cls.penalty(x)
        return inner
    
    @plotting.plot_sum
    @_sum
    @log.energy_term_log
    @plotting.plot_terms
    def energy_func(self, x):
        self.renderer.set_param(x)
        img = self.renderer.acquire_snapshot()
        self._iter_callback(*self._iter_callback_args)
        return [(func(img), weight, name) for func, name, weight in self._energy_list]

    '''
    energy_dic is a static attribute of optimizer, which maps energy
    function names to a callable instance. The mapped callable is a closure
    that takes the target image as input, and return a method calculating the
    energy function value based on input image.
    '''
    energy_dic = {
                "XOR comparison": _xor_closure,
                "first moments (normalized)": _sq_diff_closure(_get_fst_moments),
                "secondary moments (normalized)": _sq_diff_closure(_get_sec_moments)
            }
    ### end of Optimizer ###


class Renderer(Thread):
    @classmethod
    def shadow_proj_mat(cls, plane_normal, plane_point, light_pos):
        if type(plane_normal) == vec3:
            plane_normal = plane_normal.normalize()
        elif type(plane_normal) == np.array:
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
        else:
            raise TypeError("What the hell did you put in here as a normal??")
        n_t = np.array(plane_normal).reshape((1, 3))
        L = np.array(light_pos).reshape((3, 1))
        D = -plane_normal * plane_point
        ntL = np.dot(n_t, L)
        shad_mat = np.identity(4, float)
        shad_mat[0:3, 0:3] = L.dot(n_t) - (D + ntL) * np.identity(3)
        shad_mat[0:3, 3:4] = (D + ntL) * L - L * ntL
        shad_mat[3:4, 0:3] = n_t
        shad_mat[3:4, 3:4] = -ntL
        return mat4(shad_mat.astype(np.float32).T.tolist())

    class _Item:  # a temporary holder of attributes and uniforms
        pass
    
    class _Model: # a holder of obj, buffers and
        pass

    class _Window:  # a windows attribute holder
        pass

    def init(self):
        # windows initialization
        self.window = self._Window()
        self.window.width, self.window.height = (640, 480)
        self.window.handle = glfw.create_window(self.window.width * 2, self.window.height, "scene", None, None)
        if self.window.handle == None:
            glfw.terminate()
            raise RuntimeError("GLFW cannot create a window.")
        glfw.set_window_pos(self.window.handle, 9, 36)
        glfw.make_context_current(self.window.handle)
        glClearColor(0.0, 0.0, 0.2, 1.0)

        # default blinn-phong shader loading
        self.program_handle = tools.load_program("../shader/standardShading.v.glsl",
                                        "../shader/standardShading.f.glsl")
        glUseProgram(self.program_handle)

        self.cube = self._Item()
        self.cube.obj = Object("../obj/cube.obj")

        # initialize VAO
        self.vao_handle = glGenVertexArrays(1)
        glBindVertexArray(self.vao_handle)

        # bind buffers
        # indices buffer
        self.cube.i_buffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube.i_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     (c_ushort * len(self.cube.obj.indices))(*self.cube.obj.indices),
                     GL_STATIC_DRAW)
        # vertices buffer
        self.cube.v_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        v_flatten = self.flatten(self.cube.obj.vertices)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(v_flatten))(*v_flatten),
                     GL_STATIC_DRAW)
        # normals buffer
        self.cube.n_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.n_buffer)
        n_flatten = self.flatten(self.cube.obj.normals)
        glBufferData(GL_ARRAY_BUFFER,
                     (c_float * len(n_flatten))(*n_flatten),
                     GL_STATIC_DRAW)

        # attributes initializing
        self.cube.vert_loc = glGetAttribLocation(self.program_handle,
                                                 "vertexPosition_modelspace")
        glEnableVertexAttribArray(self.cube.vert_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        glVertexAttribPointer(self.cube.vert_loc, 3, GL_FLOAT, GL_FALSE, 0, None)
        # TODO: fix the existing attribute unable to retrieve problem
        self.cube.norm_loc = glGetAttribLocation(self.program_handle,
                                                 "vertexNormal_modelspace")
        glEnableVertexAttribArray(self.cube.norm_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.n_buffer)
        glVertexAttribPointer(self.cube.norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # cube uniforms
        self.cube.position = vec3(0.5, 0, 1)
        self.cube.model_mat = mat4(1.0)
        self.cube.model_mat.translate(self.cube.position)
        self.cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
        self.cube.model_mat.scale(vec3(0.5))

        # camera initializing
        self.cam_obs = self._Item()  # the camera for human observation
        self.cam_obs.view_mat = self.look_at(vec3(-1, 2, 5),
                                   vec3(0, 0, 0),
                                   vec3(0, 1, 0))
        self.cam_cap = self._Item()  # the camera to capture shadow
        self.cam_cap.view_mat = self.look_at(vec3(0, 4, 0),
                                   vec3(0, 0, 0),
                                   vec3(0, 0, -1))
        self.cam_cap.proj_mat = self.cam_obs.proj_mat = mat4.perspective(45, 4. / 3, 0.1, 100)
        self.model_view_inv = mat4(1.0) 
        self.model_view_inv = (self.cam_obs.view_mat * self.cube.model_mat).inverse()
    #     light_pos = vec3(2,1,0)
    #     light_pos = vec3(2,2,2)
        self.light_pos = vec3(3, 3, 3)
        self.cam_obs.V_loc = glGetUniformLocation(self.program_handle, "V")
        glUniformMatrix4fv(self.cam_obs.V_loc, 1, GL_FALSE, self.cam_obs.view_mat.toList())
        self.light_pos_loc = glGetUniformLocation(self.program_handle,
                                                  "LightPosition_worldspace")
        glUniform3f(self.light_pos_loc,
                    self.light_pos.x, self.light_pos.y, self.light_pos.z)
        self.cube.MVP_loc = glGetUniformLocation(self.program_handle, "MVP")
        self.cube.MVP = mat4(1)
        self.cube.M_loc = glGetUniformLocation(self.program_handle, "M")
        self.MVint_loc = glGetUniformLocation(self.program_handle, "MVint")

        # init the floor
        self.floor = self._Item()
        self.floor.model_mat = mat4.translation((0, -0.51, 0)) * mat4.scaling((5, 0.1, 5))
        self.floor.MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * self.floor.model_mat
        self.floor.MVinv = (self.cam_obs.view_mat * self.floor.model_mat).inverse()

        # initialize shadow projection program
        self.shadow_program_handle = tools.load_program("../shader/shadowProjectionShading.v.glsl",
                                                   "../shader/shadowProjectionShading.f.glsl")
        glUseProgram(self.shadow_program_handle)
        self.shadow = self._Item()
        self.shadow.MsVP_loc = glGetUniformLocation(self.shadow_program_handle, "MsVP")
        self.shadow.VP_mat = self.cam_obs.proj_mat * self.cam_obs.view_mat;
        self.shadow.VP_mat_top = self.cam_cap.proj_mat * self.cam_cap.view_mat;
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), vec3(0, -0.45, 0), self.light_pos)
        glUniform3f(glGetUniformLocation(self.shadow_program_handle, "shadowColor"),
                     0.0, 0.0, 0.0)  # black shadow
        self.shadow.v_loc = glGetAttribLocation(self.shadow_program_handle, "coord3d")
        glEnableVertexAttribArray(self.shadow.v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        glVertexAttribPointer(self.shadow.v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # init the shader to draw the basic shadow
        self.basic_program_handle = tools.load_program("../shader/basic.v.glsl",
                                                   "../shader/basic.f.glsl")
        glUseProgram(self.basic_program_handle)
        self.basic_mvp_loc = glGetUniformLocation(self.basic_program_handle, "mvp")
        self.floor_basic_mvp = self.cam_cap.proj_mat * self.cam_cap.view_mat * self.floor.model_mat
        glUniformMatrix4fv(self.basic_mvp_loc, 1, GL_FALSE, self.floor_basic_mvp.toList())
        self.basic_v_loc = glGetAttribLocation(self.basic_program_handle, "coord3d")
        glEnableVertexAttribArray(self.basic_v_loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube.v_buffer)
        glVertexAttribPointer(self.basic_v_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # initializing other stuff
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        
        _init_X_Y(self.window.width, self.window.height)
        self._X = np.arange(self.window.width).reshape(1,self.window.width)
        self._Y = np.arange(self.window.height).reshape(self.window.height,1)
    
    # obsolete
    def set_target_image(self, filepath):
        self.image_target = Image.open(filepath)
        self.image_target = self.image_target.convert("L")
        # TODO: process image if unmatch with

    def __init__(self):
        Thread.__init__(self)
        if not glfw.init():
            raise RuntimeError("Cannot start up GLFW")
        self.flatten = lambda l: [u for t in l for u in t]
        self.c_array = lambda c_type: lambda l: (c_type * len(l))(*l)
        # TODO: rewrite this stupid expedient look_at function
        self.look_at = lambda eye, at, up: mat4.lookAt(eye, 2 * eye - at, up).inverse()
#         self.init()
        self.ss_update = Lock() # ss is short for snapshot
        self.ss_ready = Lock()
        self.ss_ready.acquire()
        self.snapshot = None
        self.param_lock = Lock()
        self._init_finished_lock = Lock()
        self._init_finished_lock.acquire()
        self._models = []

    def draw(self, x):
#         self.set_param(x)
        # Render here
        # Make the window's context current
        self.shadow.shaject_mat = self.shadow_proj_mat(vec3(0, 1, 0), 
                                                       vec3(0, -0.45, 0), 
                                                       self.light_pos)

        glfw.make_context_current(self.window.handle)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, self.window.width, self.window.height)
        glUseProgram(self.program_handle)
        glUniform3f(self.light_pos_loc, *self.light_pos)
        model_view_inv = (self.cam_obs.view_mat * self.cube.model_mat).inverse()
        self.model_view_inv = (self.cam_obs.view_mat * self.cube.model_mat).inverse()
        glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, self.model_view_inv.toList())
        glUniformMatrix4fv(self.cube.M_loc, 1, GL_FALSE, self.cube.model_mat.toList())
        self.cube.MVP = self.cam_obs.proj_mat * self.cam_obs.view_mat * self.cube.model_mat
        glUniformMatrix4fv(self.cube.MVP_loc, 1, GL_FALSE, self.cube.MVP.toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None);
        glUniformMatrix4fv(self.MVint_loc, 1, GL_TRUE, self.floor.MVinv.toList())
        glUniformMatrix4fv(self.cube.M_loc, 1, GL_FALSE, self.floor.model_mat.toList())
        glUniformMatrix4fv(self.cube.MVP_loc, 1, GL_FALSE, self.floor.MVP.toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glDisable(GL_CULL_FACE)
        glUseProgram(self.shadow_program_handle)
        glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
               (self.shadow.VP_mat * self.shadow.shaject_mat * self.cube.model_mat).toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)


        glViewport(self.window.width, 0, self.window.width, self.window.height)

        glDisable(GL_CULL_FACE)
        glUseProgram(self.basic_program_handle)
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        glUseProgram(self.shadow_program_handle)
        glUniformMatrix4fv(self.shadow.MsVP_loc, 1, GL_FALSE,
               (self.shadow.VP_mat_top * self.shadow.shaject_mat * self.cube.model_mat).toList())

#         glUniformMatrix4fv(shadow_M_loc, 1, GL_FALSE, model_mat.toList())
#         glUniformMatrix4fv(shadow_VP_loc, 1, GL_FALSE, VP_mat_top.toList())
        glDrawElements(GL_TRIANGLES, len(self.cube.obj.indices),
                        GL_UNSIGNED_SHORT, None)
        # Swap front and back buffers
        glfw.swap_buffers(self.window.handle)
        if self.ss_update.locked():
            self._save_snapshot()
            self.ss_ready.release()
        glfw.poll_events()

    # obsolete method
    def optimize(self, x=None):
        if x == None:
            x = self.cube.position
#         res = optimize.minimize(fun=self._optim_obj_sec_moment, x0=x, method='BFGS', 
#                                 callback=None, bounds=((0, 2.5), (0, 2.5), (None, None)),
#                                 jac=self._get_jac(self._optim_obj_sec_moment, 0.005, x))
#         x_res = res.x
        res = cma.fmin(objective_function=self._optim_obj_sec_moment, 
             x0=x,
             sigma0=1)    
        x_res = res[0]
        self.set_param(x_res)
        print "optm ends"
    
    # obsolete method
    def _get_jac(self, func, delta, x0):
        # let func be the energy-function and delta as the uniform delta for gradient
        len_x = len(x0)
        def jac(x):
            fx = func(x)
            grad = np.zeros(len_x)        
            for i in range(len_x):
                x_t = np.zeros(len_x)
                x_t[i] = delta
                fx_t = func(x+x_t)
                grad[i] = fx_t - fx
            return grad / delta
        return jac
    
    def acquire_snapshot(self):
        self.ss_update.acquire()
        self.ss_ready.acquire()
        self.ss_update.release()
        return self.snapshot
    
    # obsolete method
    def _optim_obj_sec_moment(self, x):
        self.set_param(x)
        self.ss_update.acquire()
        self.ss_ready.acquire()
        self.ss_update.release()
        image = self.snapshot
        M_2 = self._get_moments(image)
        res = ((self.Mt_2 - M_2)**2).sum()
        print res, x
        return res
    
    def _save_snapshot(self):
        glfw.swap_buffers(self.window.handle)
        buff = glReadPixels(self.window.width, 0, self.window.width, 
                         self.window.height, GL_RGB, GL_UNSIGNED_BYTE)
        glfw.swap_buffers(self.window.handle)
        im = Image.fromstring(mode="RGB", data=buff, 
                              size=(self.window.width, self.window.height))
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.convert("L")
        self.snapshot = im
    
    # obsolete method
    def _get_moments(self, image):
        # image should be a gray scale Image object
        img = 1 - np.array(image.getdata()) / 128 # turn white to 0 and black to 1
        # using 128 in case of gray
        img = img.astype(np.int8)
        img = img.reshape(self.window.height, self.window.width)
        M_00 = float(img.sum())        
        M_10 = (self._X * img).sum()
        M_01 = (img * self._Y).sum()
        m_10 = M_10 / M_00 if M_00 else 0
        m_01 = M_01 / M_00 if M_00 else 0
        X_offset = self._X-m_10
        Y_offset = self._Y-m_01
        m_20 = ((X_offset**2)*img).sum() / M_00 if M_00 else 0
        m_02 = (img*(Y_offset**2)).sum() / M_00 if M_00 else 0
        m_11 = (X_offset*img*Y_offset).sum() / M_00 if M_00 else 0
        return np.array([m_10, m_01]), np.array([m_20, m_11, m_02])

    def set_param(self, x):
        self.param_lock.acquire()
        self.cube.model_mat = mat4(1.0)
        self.cube.model_mat.translate((x[0], x[1], x[2]))
        self.cube.model_mat.rotate(pi / 3, vec3(1.0, 0.5, 1.7))
        self.cube.model_mat.scale(vec3(0.5))
#         self.cube.model_mat.rotate(x[5], vec3(cos(x[4])*cos(x[3]), sin(x[4]), cos(x[4])*sin(x[3])))

        self.cube.position = vec3(*x)
        self.param_lock.release()
        return
    
    def set_param_indiv(self, x_i, i):
        # set one individual component of the parameters
        x = self.get_param()
        x[i] = x_i
        self.set_param(x)
    
    def get_param(self):
        return np.array(self.cube.position)

    def to_close():
        return False
    
    def wait_till_init(self):
        if self._init_finished_lock.locked(): # init is ongoing 
            self._init_finished_lock.acquire()
            self._init_finished_lock.release()
        return

    def run(self):
#         draw_projected_shadows()
        
        self.init()
        self._init_finished_lock.release()
#         self.optimize()
        while not glfw.window_should_close(self.window.handle):
            self.param_lock.acquire()
            self.draw(None)
            self.param_lock.release()
        glfw.terminate()
        pass

def _main():
    renderer = Renderer()
    renderer.start()
    gui = MyGUI(renderer)
#     gui = Weight_Dialog(['a','b','c', 'd','e','f','g'], np.array([0.03, 0.2, 0.2, 0.1, 0.05, 0.4, 0.02]))
    gui.run()
    return

if __name__ == "__main__":
    print "----------start of main---------"
    _main()
    print "---------end of main-------------"

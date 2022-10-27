import pyvjoy


XBOX_TO_PYVJOY = {
  'AXES': {
    'LS_X': 48,
    'LS_Y': 49,
    'LT': 50,
    'RS_X': 51,
    'RS_Y': 52,
    'RT': 53
  },
  'BUTTONS': {
    'A': 1,
    'B': 2,
    'X': 3,
    'Y': 4,
    'LB': 5,
    'RB': 6,
    'BACK': 7,
    'START': 8
  }
}


class PyvJoyXboxController:
  MAX_AXIS_VALUE = 0x8000
  DEFAULT_AXIS_VALUE = -1

  def __init__(self, data_key_labels, button_threshold=0.5):
    self.controller = pyvjoy.VJoyDevice(1)
    
    self.data_key_labels = data_key_labels
    self.button_threshold = button_threshold

  def emit_keys(self, output_values):
    for idx, value in enumerate(output_values):
      key_label = self.data_key_labels[idx]

      if key_label in XBOX_TO_PYVJOY['AXES'].keys():
        self.set_axis(key_label, value)
      
      elif key_label in XBOX_TO_PYVJOY['BUTTONS'].keys():
        self.set_button(key_label, value)

  def set_axis(self, label, value):
    scaled = self.scale_axis(value)
    self.controller.set_axis(XBOX_TO_PYVJOY['AXES'][label], scaled)
  
  def scale_axis(self, value):
    return int(((float(value) + 1.0) / 2.0) * self.MAX_AXIS_VALUE)
  
  def set_button(self, label, value):
    thresholded = self.threshold_button(value)
    self.controller.set_button(XBOX_TO_PYVJOY['BUTTONS'][label], thresholded)
  
  def threshold_button(self, value):
    if value >= self.button_threshold:
      return 1
    return 0

  def reset_controller(self):
    for label in XBOX_TO_PYVJOY['AXES'].keys():
      self.set_axis(label, self.DEFAULT_AXIS_VALUE)
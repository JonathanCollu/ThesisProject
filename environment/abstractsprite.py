import numpy as np
import collections
from spriteworld import constants
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
from shapely.geometry import LineString, Point, Polygon

FACTOR_NAMES = (
    'x',  # x-position of sprite center-of-mass (float)
    'y',  # y-position of sprite center-of-mass (float)
    'shape',  # shape (string)
    'angle',  # angle in degrees (scalar)
    'scale',  # size of sprite (float)
    'c0',  # first color component (scalar)
    'c1',  # second color component (scalar)
    'c2',  # third color component (scalar)
    'x_vel',  # x-component of velocity (float)
    'y_vel',  # y-component of velocity (float)
)

class AbstractSprite(object):

  def __init__(self,
               x=0.5,
               y=0.5,
               shape='square',
               angle=0,
               scale=0.1,
               c0=0,
               c1=0,
               c2=0,
               x_vel=0.0,
               y_vel=0.0):
    """Construct sprite.

    This class is agnostic to the color scheme, namely (c1, c2, c3) could be in
    RGB coordinates or HSV, HSL, etc. without this class knowing. The color
    scheme conversion for rendering must be done in the renderer.

    Args:
      x: Float in [0, 1]. x-position.
      y: Float in [0, 1]. y-position.
      shape: String. Shape of the sprite. Must be a key of constants.SHAPES.
      angle: Int. Angle in degrees.
      scale: Float in [0, 1]. Scale of the sprite, from a point to the area of
        the entire frame. This scales linearly with respect to sprite width,
        hence with power 1/2 with respect to sprite area.
      c0: Scalar. First coordinate of color.
      c1: Scalar. Second coordinate of color.
      c2: Scalar. Third coordinate of color.
      x_vel: Float. x-velocity.
      y_vel: Float. y-velocity.
    """
    self._position = np.array([x, y])
    self._shape = shape
    self._angle = angle
    self._scale = scale
    self._color = (c0, c1, c2)
    self._velocity = (x_vel, y_vel)
    self.MIN = 0
    self.MAX = 0

    # geometrical property of the shape (e.g radius or side)
    self.prop = None
    self._reset_centered_path()

  def _reset_centered_path(self):
    path = mpl_path.Path(constants.SHAPES[self._shape])
    scale_rotate = (
        mpl_transforms.Affine2D().scale(self._scale) +
        mpl_transforms.Affine2D().rotate_deg(self._angle))
    self._centered_path = scale_rotate.transform_path(path)

  ################## Handle Overlaps ####################################

  def _circle_circle_overlap(self, other, direction):
    line1 = LineString([self.position, other.position])
    p1 = line1.intersection(self.polygon)
    p1 = Point([p for p in list(p1.coords) if Point(p) != Point(self.position)][0])
    if direction == "down":
      line2 = LineString([p1, (p1.x, 1)])
      p2 = line2.intersection(other.polygon)
      if p2.__class__.__name__ == "LineString" and list(p2.coords):
        p2 = Point([p for p in list(p2.coords) if Point(p) != p1][0])
        self._position[1] += p2.y - p1.y 
    elif direction == "up":
      line2 = LineString([(p1.x, 0), p1])
      p2 = line2.intersection(other.polygon)
      if p2.__class__.__name__ == "LineString" and list(p2.coords):
        p2 = Point([p for p in list(p2.coords) if Point(p) != p1][0])
        self._position[1] -= p1.y - p2.y
    elif direction == "right":
      line2 = LineString([(0, p1.y), p1])
      p2 = line2.intersection(other.polygon)            
      if p2.__class__.__name__ == "LineString" and list(p2.coords):
        p2 = Point([p for p in list(p2.coords) if Point(p) != p1][0])
        self._position[0] -= p1.x - p2.x
    else:
      line2 = LineString([p1, (1, p1.y)])
      p2 = line2.intersection(other.polygon)
      if p2.__class__.__name__ == "LineString" and list(p2.coords):
        p2 = Point([p for p in list(p2.coords) if Point(p) != p1][0])
        self._position[0] += p2.x - p1.x

  def _circle_square_overlap(self, other, direction):
    circle = self.contours
    if direction == "down":
      if other.vertices[1][0] > self.x:
        line1 = LineString([other.vertices[1], other.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[1] += other.vertices[1][1] - p.y
        except: pass
      elif other.vertices[0][0] < self.x:
        line1 = LineString([other.vertices[0], other.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[1] += other.vertices[0][1] - p.y
        except: pass
      else: 
        self._position[1] += other.bounds[3] - (self.y - self.prop) + 1e-5
    elif direction == "up":
      if other.vertices[2][0] > self.x:
        line1 = LineString([other.vertices[1], other.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[1] -= p.y - other.vertices[2][1]
        except: pass
      elif other.vertices[3][0] < self.x:
        line1 = LineString([other.vertices[0], other.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[1] -= p.y - other.vertices[2][1]
        except: pass
      else: 
        self._position[1] -= (self.y + self.prop) - other.bounds[1] + 1e-5
    elif direction == "right":
      if other.vertices[1][1] < self.y:
        line1 = LineString([other.vertices[0], other.vertices[1]])
        p = line1.intersection(circle)
        try: self._position[0] -= p.x - other.vertices[1][0]
        except: pass
      elif other.vertices[2][1] > self.y:
        line1 = LineString([other.vertices[2], other.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[0] -= p.x - other.vertices[2][0]
        except: pass
      else:
        self._position[0] -= (self.x + self.prop) - other.bounds[0]  + 1e-5
    else:
      if other.vertices[0][1] < self.y:
        line1 = LineString([other.vertices[0], other.vertices[1]])
        p = line1.intersection(circle)
        try: self._position[0] += other.vertices[0][0] - p.x
        except: pass
      elif other.vertices[3][1] > self.y:
        line1 = LineString([other.vertices[2], other.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[0] += other.vertices[3][0] - p.x
        except: pass
      else:
        self._position[0] += other.bounds[2] - (self.x - self.prop)  + 1e-5

  def _square_circle_overlap(self, other, direction):
    circle = other.contours
    if direction == "down":
      if self.vertices[3][0] < other.x:
        line1 = LineString([self.vertices[0], self.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[1] += p.y - self.vertices[3][1]
        except: pass
      elif self.vertices[2][0] > other.x:
        line1 = LineString([self.vertices[1], self.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[1] += p.y - self.vertices[2][1]
        except: pass
      else: self._position[1] += (other.y + other.prop) - self.bounds[1] + 1e-5
    elif direction == "up":
      if self.vertices[1][0] > other.x:
        line1 = LineString([self.vertices[1], self.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[1] -= self.vertices[1][1] - p.y
        except: pass
      elif self.vertices[0][0] < other.x:
        line1 = LineString([self.vertices[0], self.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[1] -= self.vertices[0][1] - p.y
        except: pass
      else:
        if other.bounds[3] > (other.y - other.prop):
          self._position[1] -= other.bounds[3] - (other.y - other.prop) + 1e-5
    elif direction == "right":
      if self.vertices[0][1] < other.y:
        line1 = LineString([self.vertices[0], self.vertices[1]])
        p = line1.intersection(circle)
        try: self._position[0] -= self.vertices[0][0] - p.x
        except: pass
      elif self.vertices[3][1] > other.y:
        line1 = LineString([self.vertices[2], self.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[0] -= self.vertices[3][0] - p.x
        except: pass
      else: self._position[0] -= self.bounds[2] - (other.x - other.prop) + 1e-5
    else:
      if self.vertices[1][1] < other.y:
        line1 = LineString([self.vertices[0], self.vertices[1]])
        p = line1.intersection(circle)
        try: self._position[0] += p.x - self.vertices[1][0]
        except: pass
      elif self.vertices[2][1] > other.y:
        line1 = LineString([self.vertices[2], self.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[0] += p.x - self.vertices[2][0]
        except: pass
      else: self._position[0] += (other.x + other.prop) - self.bounds[0] + 1e-5

  def _square_square_overlap(self, other, direction):
    if direction == "down":
      if other.bounds[3] >= self.bounds[1]:
        self._position[1] += other.bounds[3] - self.bounds[1] + 1e-5
    elif direction == "up":
      if self.bounds[3] >= other.bounds[1]:
        self._position[1] -= self.bounds[3] - other.bounds[1] + 1e-5
    elif direction == "right":
      if self.bounds[2] >= other.bounds[0]:  
        self._position[0] -= self.bounds[2] - other.bounds[0] + 1e-5
    else:
      if other.bounds[2] >= self.bounds[0]:
        self._position[0] += other.bounds[2] - self.bounds[0] + 1e-5

  def _circle_triangle_overlap(self, other, direction):
    r = self.prop
    if direction == "down":
      if not (self.x - r >= other.vertices[0][0] or self.x + r <= other.vertices[0][0]):
        if other.bounds[3] > self.y - r: self._position[1] += other.bounds[3] - (self.y - r)  + 1e-5
      else:
        if self.x <= other.vertices[0][0]:
          p = self.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
          line2 = LineString([other.vertices[0], other.vertices[1]])
        elif self.x > other.vertices[0][0]:
          p = self.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
          line2 = LineString([other.vertices[0], other.vertices[2]])
        line1 = LineString([p, (p[0],1)])
        line3 = LineString([other.vertices[0], (other.vertices[0][0], other.y)])
        d1, d2 = (0, 0)
        if line1.intersection(line2):
          point = line1.intersection(line2)
          d1 = point.y - p[1]  + 1e-5
        if line3.intersects(self.contours):
          d2 = other.vertices[0][1] - line3.intersection(self.contours).bounds[1] + 1e-5
        self._position[1] += max(d1, d2)   
    elif direction == "up":
      if other.vertices[2][0] < self.x:
        line1 = LineString([other.vertices[2], (other.vertices[2][0], 1)])
        point = line1.intersection(self.contours)        
        try: self._position[1] -= point.bounds[3]  - other.vertices[2][1]
        except: pass
      elif other.vertices[1][0] > self.x:
        line1 = LineString([other.vertices[1], (other.vertices[1][0], 1)])
        point = line1.intersection(self.contours)
        try: self._position[1] -= point.bounds[3]  - other.vertices[1][1] 
        except: pass
      else: self._position[1] -= other.bounds[3] - self.bounds[1]
    elif direction == "right":
      if self.y - r <= other.vertices[0][1] and self.y - r >= other.vertices[1][1]:
        p = self.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
        line1 = LineString([(0, self.y), p]) 
        line2 = LineString([other.vertices[0], other.vertices[1]])
        point = line1.intersection(line2)
        try: self._position[0] -= p[0] - point.x 
        except: pass
      else:
        line1 = LineString([other.vertices[1], other.vertices[2]])
        line2 = LineString(self.vertices)
        p = line1.intersection(line2)
        try: self._position[0] -= p.x - other.vertices[1][0]  + 1e-5
        except: pass
    else:
      if self.y - r <= other.vertices[0][1] and self.y -r >= other.vertices[2][1]:
        p = self.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
        line1 = LineString([p, (1, self.y)])
        line2 = LineString([other.vertices[0], other.vertices[2]])
        point = line1.intersection(line2)
        try: self._position[0] += point.x - p[0]
        except: pass
      else:
        line1 = LineString([other.vertices[1], other.vertices[2]])
        line2 = LineString(self.vertices)
        p = line1.intersection(line2)
        try: self._position[0] += other.vertices[2][0] - p.x  + 1e-5
        except: pass

  def _square_triangle_overlap(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      if self_vert[3][0] < other_vert[0][0]:
        line1 = LineString([other_vert[0], other_vert[1]])
        line2 = LineString([self_vert[3], (self_vert[3][0], 1)])
        point = line1.intersection(line2)
        try: self._position[1] += point.y  - self_vert[3][1]
        except: pass
      elif self_vert[2][0] > other_vert[0][0]:
        line1 = LineString([other_vert[0], other_vert[2]])
        line2 = LineString([self_vert[2], (self_vert[2][0], 1)])
        point = line1.intersection(line2)
        try: self._position[1] += point.y  - self_vert[2][1] 
        except: pass
      else: self._position[1] += other.bounds[3] - self.bounds[1]
    elif direction == "up": self._position[1] -= self.bounds[3] - other.bounds[1] + 1e-5
    elif direction == "right":
      if self.bounds[1] >= other.bounds[1]: 
        line1 = LineString([other_vert[0], other_vert[1]])
        line2 = LineString([(0, self_vert[3][1]), self_vert[3]])
        point = line1.intersection(line2)
        try: self._position[0] -= self_vert[3][0] - point.x
        except: pass
      else: self._position[0] -= self.bounds[2] - other.bounds[0]
    else:
      if self.bounds[1] >= other.bounds[1]:
        line1 = LineString([other_vert[0], other_vert[2]])
        line2 = LineString([self_vert[2], (1, self_vert[2][1])])
        point = line1.intersection(line2)
        try: self._position[0] += point.x - self_vert[2][0]
        except: pass
      else: self._position[0] += other.bounds[2] - self.bounds[0]
    del self_vert, other_vert

  def _triangle_circle_overlap(self, other, direction):
    r = other.prop
    if direction == "down":
      if self.vertices[2][0] < other.x:
        line1 = LineString([self.vertices[2], (self.vertices[2][0], 1)])
        point = line1.intersection(other.contours)        
        try: self._position[1] += point.bounds[3]  - self.vertices[2][1]
        except: pass
      elif self.vertices[1][0] > other.x:
        line1 = LineString([self.vertices[1], (self.vertices[1][0], 1)])
        point = line1.intersection(other.contours)
        try: self._position[1] += point.bounds[3]  - self.vertices[1][1] 
        except: pass
      else: self._position[1] += other.bounds[3] - self.bounds[1]
    elif direction == "up":
      if not (other.x + 0.02 <= self.vertices[0][0] or other.x - 0.02 >= self.vertices[0][0]):   
        if self.bounds[3] > other.position[1] - r:
          self.position[1] -= self.bounds[3] - (other.position[1] - r) + 1e-5
      else:
        if other.x <= self.vertices[0][0]:
          p = other.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
          line2 = LineString([self.vertices[0], self.vertices[1]])
        if other.x > self.vertices[0][0]:
          p = other.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
          line2 = LineString([self.vertices[0], self.vertices[2]])
        line1 = LineString([p, (p[0],1)])
        line3 = LineString([self.vertices[0], (self.vertices[0][0], self.y)])
        d1, d2 = (0, 0)
        if line1.intersection(line2):
          point = line1.intersection(line2)
          d1 = point.y - p[1]  + 1e-5
        if line3.intersects(other.contours):
          d2 = self.vertices[0][1] - line3.intersection(other.contours).bounds[1] + 1e-5
        self._position[1] -= max(d1, d2) # if min(d1, d2) != 1 else 0
    elif direction == "right":
      if self.vertices[2][1] >= other.y + r * np.sin((5*np.pi)/4):
        p1 = other.polygon.intersection(LineString([(0, self.vertices[2][1]), self.vertices[2]]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(self.vertices[2])]
        if p1: self._position[0] -= self.vertices[2][0] - p1[0][0]
      elif self.vertices[0][1] < other.y + r * np.sin((5*np.pi)/4):
        p1 = other.polygon.intersection(LineString([(0, self.vertices[0][1]), self.vertices[0]]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(self.vertices[0])]
        if p1: self._position[0] -= self.vertices[0][0] - p1[0][0]
      else:
        p = other.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
        line1 = LineString([p, (1, p[1])])
        line2 = LineString([self.vertices[0], self.vertices[2]])
        point = line1.intersection(line2)
        try: self._position[0] -= point.x - p[0]  + 1e-5
        except: pass
    else:
      if self.vertices[1][1] >= other.y + r * np.sin((7*np.pi)/4):
        p1 = other.polygon.intersection(LineString([self.vertices[1], (1, self.vertices[1][1])]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(self.vertices[1])]
        if p1: self._position[0] += p1[0][0] - self.vertices[1][0]
      elif self.vertices[0][1] < other.y + r * np.sin((7*np.pi)/4):
        p1 = other.polygon.intersection(LineString([self.vertices[0], (1, self.vertices[0][1])]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(self.vertices[0])]
        if p1: self._position[0] += p1[0][0] - self.vertices[0][0] + 1e-5
      else:
        p = other.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
        line1 = LineString([(0, p[1]), p])
        line2 = LineString([self.vertices[0], self.vertices[1]])
        point = line1.intersection(line2)
        try: self._position[0] += p[0] - point.x + 1e-5
        except: pass

  def _triangle_square_overlap(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down": self._position[1] += other.bounds[3] - self.bounds[1] + 1e-5
    elif direction == "up":
      if other_vert[3][0] < self_vert[0][0]:
        line1 = LineString([self_vert[0], self_vert[1]])
        line2 = LineString([other_vert[3], (other_vert[3][0], 1)])
        point = line1.intersection(line2)
        try: self._position[1] -= point.y - other_vert[3][1] + 1e-5
        except: pass
      elif other_vert[2][0] > self_vert[0][0]:
        line1 = LineString([self_vert[0], self_vert[2]])
        line2 = LineString([other_vert[2], (other_vert[2][0], 1)])
        point = line1.intersection(line2)
        try: self._position[1] -= point.y - other_vert[2][1] + 1e-5
        except: pass
      else: self._position[1] -= self.bounds[3] - other.bounds[1] + 1e-5
    elif direction == "right":
      if self.bounds[1] >= other.bounds[1]:
        self._position[0] -= self.bounds[2] - other.bounds[0] + 1e-5
      else:
        line1 = LineString([self_vert[0], self_vert[2]])
        line2 = LineString([other_vert[2], (1, other_vert[2][1])])
        point = line1.intersection(line2)
        try: self._position[0] -= point.x - other_vert[2][0] + 1e-5
        except: pass
    else:
      if self.bounds[1] >= other.bounds[1]:
        self._position[0] += other.bounds[2] - self.bounds[0] + 1e-5
      else:
        line1 = LineString([self_vert[0], self_vert[1]])
        line2 = LineString([(0, other_vert[3][1]), other_vert[3]])
        point = line1.intersection(line2)
        try: self._position[0] += other_vert[3][0] - point.x + 1e-5
        except: pass
    del self_vert, other_vert

  
  def _triangle_triangle_overlap(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      if self_vert[2][0] < other_vert[0][0]:
        line1 = LineString([self_vert[2], (self_vert[2][0], 1)])
        line2 = LineString([other_vert[0], other_vert[1]])
        point = line1.intersection(line2)
        try: self._position[1] += point.y - self_vert[2][1] 
        except: pass
      elif self_vert[1][0] > other_vert[0][0]:
        line1 = LineString([self_vert[1], (self_vert[1][0], 1)])
        line2 = LineString([other_vert[0], other_vert[2]])
        point = line1.intersection(line2)
        try: self._position[1] += point.y - self_vert[1][1]
        except: pass
      else:
        if self.bounds[1] < other.bounds[3]:
          self._position[1] += other.bounds[3] - self.bounds[1]
    elif direction == "up":
      if other_vert[2][0] < self_vert[0][0]:
        line1 = LineString([other_vert[2], (other_vert[2][0], 1)])
        line2 = LineString([self_vert[0], self_vert[1]])
        point = line1.intersection(line2)
        try: self._position[1] -= point.y - other_vert[2][1] 
        except: pass
      elif other_vert[1][0] > self_vert[0][0]:
        line1 = LineString([other_vert[1], (other_vert[1][0], 1)])
        line2 = LineString([self_vert[0], self_vert[2]])
        point = line1.intersection(line2)
        try: self._position[1] -= point.y- other_vert[1][1]
        except: pass
      else:
        if other.bounds[1] < self.bounds[3]:
          self._position[1] -= self.bounds[3] - other.bounds[1]
    elif direction == "right":
      if self.bounds[1] > other.bounds[1]:
        line1 = LineString([(0, self_vert[2][1]), self_vert[2]])
        line2 = LineString([other_vert[0], other_vert[1]])
        point = line1.intersection(line2)
        try: self._position[0] -= self_vert[2][0] - point.x
        except: pass
      elif self.bounds[1] < other.bounds[1]:
        line1 = LineString([other_vert[1], (1, other_vert[1][1])])
        line2 = LineString([self_vert[0], self_vert[2]])
        point = line1.intersection(line2)
        try: self._position[0] -= point.x - other_vert[1][0]
        except: pass
      else:
        if other.bounds[0] < self.bounds[2]: self._position[0] -= self.bounds[2] - other.bounds[0]
    else:
      if self.bounds[1] > other.bounds[1]:
        line1 = LineString([self_vert[1], (1, self_vert[1][1])])
        line2 = LineString([other_vert[0], other_vert[2]])
        point = line1.intersection(line2)
        try: self._position[0] += point.x - self_vert[1][0]
        except: pass
      elif self.bounds[1] < other.bounds[1]:
        line1 = LineString([(0, other_vert[2][1]), other_vert[2]])
        line2 = LineString([self_vert[0], self_vert[1]])
        point = line1.intersection(line2)
        try: self._position[0] += other_vert[2][0]- point.x
        except: pass
      else:
        if other.bounds[2] > self.bounds[0]: self._position[0] += other.bounds[2] - self.bounds[0]            
    del self_vert, other_vert

  ################## Handle Collisions ####################################

  def _handle_circle_circle(self, other, direction):
    line_distance = LineString([self.position, other.position])
    c1 = self.contours
    c2 = other.contours
    # p1 is the point of c1 touching the point p2 of c2
    p1 = line_distance.intersection(c1)
    p2 = line_distance.intersection(c2)
    if direction == "down": self._position[1] -= p1.y - p2.y
    elif direction == "up": self._position[1] += p2.y - p1.y
    elif direction == "right": self._position[0] += p2.x - p1.x
    else: self._position[0] -= p1.x - p2.x


  def _handle_circle_square(self, other, direction):
    circle = self.contours
    if direction == "down":
      if other.vertices[1][0] > self.x:
        line1 = LineString([other.vertices[1], (other.vertices[1][0], self.y)])
        p = line1.intersection(circle)
        try: self._position[1] -= p.y - other.vertices[1][1]
        except: pass
      elif other.vertices[0][0] < self.x:
        line1 = LineString([other.vertices[0], (other.vertices[0][0], self.y)])
        p = line1.intersection(circle)
        try: self._position[1] -= p.y - other.vertices[0][1]
        except: pass
      else: self._position[1] -= self.y - self.prop - other.bounds[3]
    elif direction == "up":
      if other.vertices[2][0] > self.x:
        line1 = LineString([(other.vertices[2][0], self.y), other.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[1] += other.vertices[2][1] - p.y
        except: pass
      elif other.vertices[3][0] < self.x:
        line1 = LineString([(other.vertices[3][0], self.y), other.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[1] += other.vertices[3][1] - p.y
        except: pass
      else: self._position[1] += other.bounds[1] - (self.y + self.prop)
    elif direction == "right":
      if other.vertices[1][1] < self.y:
        line1 = LineString([(self.x, other.vertices[1][1]), other.vertices[1]])
        p = line1.intersection(circle)
        try: self._position[0] += other.vertices[1][0] - p.x
        except: pass
      elif other.vertices[2][1] > self.y:
        line1 = LineString([(self.x, other.vertices[2][1]), other.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[0] += other.vertices[2][0] - p.x
        except: pass
      else: self._position[0] += other.bounds[0] - (self.x + self.prop)
    else:
      if other.vertices[0][1] < self.y:
        line1 = LineString([other.vertices[0], (self.x, other.vertices[0][0])])
        p = line1.intersection(circle)
        try: self._position[0] -= p.x - other.vertices[0][0]
        except: pass
      elif other.vertices[3][1] > self.y:
        line1 = LineString([other.vertices[3], (self.x, other.vertices[3][0])])
        p = line1.intersection(circle)
        try: self._position[0] -= p.x - other.vertices[3][0]
        except: pass
      else: self._position[0] -= self.x - self.prop - other.bounds[2]

  def _handle_square_circle(self, other, direction):
    circle = other.contours
    if direction == "down":
      if self.vertices[2][0] > other.x:
        line1 = LineString([(self.vertices[2][0], other.y), self.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[1] -= self.vertices[2][1] - p.y
        except: pass
      elif self.vertices[3][0] < other.x:
        line1 = LineString([(self.vertices[3][0], other.y), self.vertices[3]])
        p = line1.intersection(circle)
        try: self._position[1] -= self.vertices[3][1] - p.y
        except: pass
      else: self._position[1] -= self.bounds[1] - (other.y + other.prop)
    elif direction == "up":
      if self.vertices[1][0] > other.x:
        line1 = LineString([self.vertices[1], (self.vertices[1][0], other.y)])
        p = line1.intersection(circle)
        try: self._position[1] += p.y - self.vertices[1][1]
        except: pass
      elif self.vertices[0][0] < other.x:
        line1 = LineString([self.vertices[0], (self.vertices[0][0], other.y)])
        p = line1.intersection(circle)
        try: self._position[1] += p.y - self.vertices[0][1]
        except: pass
      else: self._position[1] += other.y - other.prop - self.bounds[3]
    elif direction == "right":
      if self.vertices[3][1] > other.y:
        line1 = LineString([self.vertices[3], (other.x, self.vertices[3][1])])
        p = line1.intersection(circle)
        try: self._position[0] += p.x - self.vertices[3][0]
        except: pass
      elif self.vertices[0][1] < other.y:
        line1 = LineString([self.vertices[0], (other.x, self.vertices[0][1])])
        p = line1.intersection(circle)
        try: self._position[0] += p.x - self.vertices[0][0]
        except: pass
      else: self._position[0] += other.x - other.prop - self.bounds[2]
    else:
      if self.vertices[2][1] > other.y:
        line1 = LineString([(other.x, self.vertices[2][1]), self.vertices[2]])
        p = line1.intersection(circle)
        try: self._position[0] -= self.vertices[2][0] - p.x
        except: pass
      elif self.vertices[1][1] < other.y:
        line1 = LineString([(other.x, self.vertices[1][1]), self.vertices[1]])
        p = line1.intersection(circle)
        try: self._position[0] -= self.vertices[1][0] - p.x
        except: pass
      else: self._position[0] -= self.bounds[0] - (other.x + other.prop)

  def _handle_square_square(self, other, direction):
    if direction == "down":
      self._position[1] -= self.bounds[1] - other.bounds[3]
    elif direction == "up":
      self._position[1] += other.bounds[1] - self.bounds[3]
    elif direction == "right":
      self._position[0] += other.bounds[0] - self.bounds[2]
    else:
      self._position[0] -= self.bounds[0] - other.bounds[2]

  def _handle_square_triangle(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      if self_vert[3][0] < other_vert[0][0]:
        line1 = LineString([other_vert[0], other_vert[1]])
        line2 = LineString([(self_vert[3][0], 0), self_vert[3]])
        point = line1.intersection(line2)
        try: self._position[1] -= self_vert[3][1] - point.y 
        except: pass
      elif self_vert[2][0] > other_vert[0][0]:
        line1 = LineString([other_vert[0], other_vert[2]])
        line2 = LineString([(self_vert[2][0], 0), self_vert[2]])
        point = line1.intersection(line2)
        try: self._position[1] -= self_vert[2][1] - point.y 
        except: pass
      else:
        self._position[1] -= self.bounds[1] - other.bounds[3]
    elif direction == "up":
      self._position[1] += other.bounds[1] - self.bounds[3]
    elif direction == "right":
      if self.bounds[1] >= other.bounds[1]: 
        line1 = LineString([other_vert[0], other_vert[1]])
        line2 = LineString([self_vert[3], (1, self_vert[3][1])])
        point = line1.intersection(line2)
        try: self._position[0] += point.x - self_vert[3][0] 
        except: pass
      else:
        self._position[0] += other.bounds[0] - self.bounds[2]
    else:
      if self.bounds[1] >= other.bounds[1]:
        line1 = LineString([other_vert[0], other_vert[2]])
        line2 = LineString([(0, self_vert[2][1]), self_vert[2]])
        point = line1.intersection(line2)
        try: self._position[0] -= self_vert[2][0] - point.x
        except: pass
      else:
        self._position[0] -= self.bounds[0] - other.bounds[2]
    del self_vert, other_vert

  def _handle_triangle_square(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      self._position[1] -= self.bounds[1] - other.bounds[3]
    elif direction == "up":
      if other_vert[3][0] < self_vert[0][0]:
        line1 = LineString([self_vert[0], self_vert[1]])
        line2 = LineString([(other_vert[3][0], 0), other_vert[3]])
        point = line1.intersection(line2)
        try: self._position[1] += other_vert[3][1] - point.y 
        except: pass
      elif other_vert[2][0] > self_vert[0][0]:
        line1 = LineString([self_vert[0], self_vert[2]])
        line2 = LineString([(other_vert[2][0], 0), other_vert[2]])
        point = line1.intersection(line2)
        try: self._position[1] += other_vert[2][1] - point.y 
        except: pass
      else:
        self._position[1] += other.bounds[1] - self.bounds[3]
    elif direction == "right":
      if self.bounds[1] >= other.bounds[1]:
        self._position[0] += other.bounds[0] - self.bounds[2]
      else:
        line1 = LineString([self_vert[0], self_vert[2]])
        line2 = LineString([(0, other_vert[2][1]), other_vert[2]])
        point = line1.intersection(line2)
        try: self._position[0] += other_vert[2][0] - point.x 
        except: pass
    else:
      if self.bounds[1] >= other.bounds[1]:
        self._position[0] -= self.bounds[0] - other.bounds[2]
      else:
        line1 = LineString([self_vert[0], self_vert[1]])
        line2 = LineString([other_vert[3], (1, other_vert[3][1])])
        point = line1.intersection(line2)
        try: self._position[0] -= point.x - other_vert[3][0] 
        except: pass
    del self_vert, other_vert

  def _handle_circle_triangle(self, other, direction):
    r = self.prop
    if direction == "down":
      if self.y - r >= other.vertices[0][1] and self.x - other.vertices[0][0] <= 0.02:
        self._position[1] -= self.y - r - other.bounds[3]
      else:  
        if self.x <= other.vertices[0][0]:
          p = self.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
          line2 = LineString([other.vertices[0], other.vertices[1]])
        if self.x > other.vertices[0][0]:
          p = self.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
          line2 = LineString([other.vertices[0], other.vertices[2]])
        line1 = LineString([(p[0],0), p])
        line3 = LineString([other.vertices[0], (other.vertices[0][0], self.y)])
        d1, d2 = (0, 0)
        if line1.intersection(line2):
          point = line1.intersection(line2)
          d1 = p[1] - point.y  
        if line3.intersects(self.contours):
          d2 = line3.intersection(self.contours).y - other.vertices[0][1]
        self._position[1] -= max(d1, d2) #if min(d1, d2) != 1 else 0
         
    elif direction == "up":
      self._position[1] += other.bounds[1] - r - self.y
    elif direction == "right":
      if self.y - r <= other.vertices[0][1] and self.y - r >= other.vertices[1][1]:
        p = self.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
        line1 = LineString([p, (1, self.y)])
        line2 = LineString([other.vertices[0], other.vertices[1]])
        point = line1.intersection(line2)
        try: self._position[0] += point.x - p[0]
        except: pass 
      else:
        line1 = LineString([(self.x, other.vertices[1][1]), other.vertices[1]])
        line2 = LineString(self.vertices)
        p = line1.intersection(line2)
        try: self._position[0] += other.vertices[1][0] - p.x
        except: pass
    else:
      if self.y - r <= other.vertices[0][1] and self.y - r >= other.vertices[1][1]:
        p = self.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
        line1 = LineString([(0, self.y), p])
        line2 = LineString([other.vertices[0], other.vertices[2]])
        point = line1.intersection(line2)
        try: self._position[0] -= p[0] - point.x
        except: pass 
      else:
        line1 = LineString([other.vertices[2], (self.x, other.vertices[2][1])])
        line2 = LineString(self.vertices)
        p = line1.intersection(line2)
        try: self._position[0] -= p.x - other.vertices[2][0] 
        except: pass

  def _handle_triangle_circle(self, other, direction):
    r = other.prop
    if direction == "down": 
      if self.vertices[2][0] < other.x:
        line1 = LineString([self.vertices[2], (self.vertices[2][0], other.y)])
        point = line1.intersection(other.contours)
        try: self._position[1] -= self.vertices[2][1] - point.bounds[3]
        except: pass
      elif self.vertices[1][0] > other.x:
        line1 = LineString([self.vertices[1], (self.vertices[1][0], other.y)])
        point = line1.intersection(other.contours)
        try: self._position[1] -= self.vertices[1][1] - point.bounds[3]
        except: pass
      else: self._position[1] -= self.bounds[1] - other.bounds[3]
    elif direction == "up":
      if not (other.x + 0.02 <= self.vertices[0][0] or other.x - 0.02 >= self.vertices[0][0]):
        self._position[1] += other.y - r - self.bounds[3]
      else:
        if other.x <= self.vertices[0][0]:
          p = other.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
          line2 = LineString([self.vertices[0], self.vertices[1]])
        if other.x > self.vertices[0][0]:
          p = other.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
          line2 = LineString([self.vertices[0], self.vertices[2]])
        line1 = LineString([(p[0],0), p])
        line3 = LineString([self.vertices[0], (self.vertices[0][0], other.y)])
        d1, d2 = (0, 0)
        if line1.intersection(line2):
          point = line1.intersection(line2)
          d1 = p[1] - point.y  
        if line3.intersects(other.contours):
          d2 = line3.intersection(other.contours).y - self.vertices[0][1]
        self._position[1] += max(d1, d2) # if min(d1, d2) != 1 else 0
    elif direction == "right":
      if self.vertices[2][1] >= other.y + r * np.sin((5*np.pi)/4):
        p1 = other.polygon.intersection(LineString([self.vertices[2], other.position]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: self._position[0] += p1[0][0] - self.vertices[2][0]
      elif self.vertices[0][1] < other.y + r * np.sin((5*np.pi)/4):
        p1 = other.polygon.intersection(LineString([self.vertices[2], other.position]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: self._position[0] += p1[0][0] - self.vertices[0][0]
      else:
        p = other.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
        line1 = LineString([(0, p[1]), p])
        line2 = LineString([self.vertices[0], self.vertices[2]])
        point = line1.intersection(line2)
        try: self._position[0] += p[0] - point.x
        except: pass
    else:
      if self.vertices[1][1] >= other.y + r * np.sin((7*np.pi)/4):
        p1 = other.polygon.intersection(LineString([other.position, self.vertices[1]]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: self._position[0] -= self.vertices[1][0] - p1[0][0]
      elif self.vertices[0][1] < other.y + r * np.sin((7*np.pi)/4):
        p1 = other.polygon.intersection(LineString([other.position, self.vertices[0]]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: self._position[0] -= self.vertices[0][0] - p1[0][0]
      else:
        p = other.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
        line1 = LineString([p, (1, p[1])])
        line2 = LineString([self.vertices[0], self.vertices[1]])
        point = line1.intersection(line2)
        try: self._position[0] -= point.x - p[0]
        except: pass

  def _handle_triangle_triangle(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      if self_vert[2][0] < other_vert[0][0]:
        line1 = LineString([(self_vert[2][0], 0), self_vert[2]])
        line2 = LineString([other_vert[0], other_vert[1]])
        point = line1.intersection(line2)
        try: self._position[1] -= self_vert[2][1] - point.y
        except: pass
      elif self_vert[1][0] > other_vert[0][0]:
        line1 = LineString([(self_vert[1][0], 0), self_vert[1]])
        line2 = LineString([other_vert[0], other_vert[2]])
        point = line1.intersection(line2)
        try: self._position[1] -= self_vert[1][1] - point.y
        except: pass
      else:
        self._position[1] -= self.bounds[1] - other.bounds[3]
    elif direction == "up":
      if other_vert[2][0] < self_vert[0][0]:
        line1 = LineString([(other_vert[2][0], 0), other_vert[2]])
        line2 = LineString([self_vert[0], self_vert[1]])
        point = line1.intersection(line2)
        try: self._position[1] += other_vert[2][1] - point.y
        except: pass
      elif other_vert[1][0] > self_vert[0][0]:
        line1 = LineString([(other_vert[1][0], 0), other_vert[1]])
        line2 = LineString([self_vert[0], self_vert[2]])
        point = line1.intersection(line2)
        try: self._position[1] += other_vert[1][1] - point.y
        except: pass
      else:
        self._position[1] += other.bounds[1] - self.bounds[3]
    elif direction == "right":
      if self.bounds[1] > other.bounds[1]:
        line1 = LineString([self_vert[2], (1, self_vert[2][1])])
        line2 = LineString([other_vert[0], other_vert[1]])
        point = line1.intersection(line2)
        try: self._position[0] += point.x - self_vert[2][0]
        except: pass
      elif self.bounds[1] < other.bounds[1]:
        line1 = LineString([(0, other_vert[1][1]), other_vert[1]])
        line2 = LineString([self_vert[0], self_vert[2]])
        point = line1.intersection(line2)
        try: self._position[0] += other_vert[1][0] - point.x
        except: pass
      else:
        self._position[0] += other.bounds[0] - self.bounds[2]
    else:
      if self.bounds[1] > other.bounds[1]:
        line1 = LineString([(0, self_vert[1][1]), self_vert[1]])
        line2 = LineString([other_vert[0], other_vert[2]])
        point = line1.intersection(line2)
        try: self._position[0] -= self_vert[1][0] - point.x
        except: pass
      elif self.bounds[1] < other.bounds[1]:
        line1 = LineString([other_vert[2], (1, other_vert[2][1])])
        line2 = LineString([self_vert[0], self_vert[1]])
        point = line1.intersection(line2)
        try: self._position[0] -= point.x - other_vert[2][0]
        except: pass
      else:
        self._position[0] -= other.bounds[2] - self.bounds[0]
    del self_vert, other_vert

  ##### DISTANCES ##################################
  def circle_circle_distance(self, other, direction):
    line_distance = LineString([self.position, other.position])
    c1 = self.contours
    c2 = other.contours
    # p1 is the point of c1 touching the point p2 of c2
    p1 = line_distance.intersection(c1)
    p2 = line_distance.intersection(c2)
    if direction == "down": return p1.y - p2.y
    elif direction == "up": return p2.y - p1.y
    elif direction == "right": return p2.x - p1.x
    else: return p1.x - p2.x


  def circle_square_distance(self, other, direction):
    circle = self.contours
    if direction == "down":
      if other.vertices[1][0] > self.x:
        line1 = LineString([other.vertices[1], (other.vertices[1][0], self.y)])
        p = line1.intersection(circle)
        try: return p.y - other.vertices[1][1]
        except: return 1
      elif other.vertices[0][0] < self.x:
        line1 = LineString([other.vertices[0], (other.vertices[0][0], self.y)])
        p = line1.intersection(circle)
        try: return p.y - other.vertices[0][1]
        except: return 1
      else: return self.y - self.prop - other.bounds[3]
    elif direction == "up":
      if other.vertices[2][0] > self.x:
        line1 = LineString([(other.vertices[2][0], self.y), other.vertices[2]])
        p = line1.intersection(circle)
        try: return other.vertices[2][1] - p.y
        except: return 1
      elif other.vertices[3][0] < self.x:
        line1 = LineString([(other.vertices[3][0], self.y), other.vertices[3]])
        p = line1.intersection(circle)
        try: return other.vertices[3][1] - p.y
        except: return 1
      else: return other.bounds[1] - (self.y + self.prop)
    elif direction == "right":
      if other.vertices[1][1] < self.y:
        line1 = LineString([(self.x, other.vertices[1][1]), other.vertices[1]])
        p = line1.intersection(circle)
        try: return other.vertices[1][0] - p.x
        except: return 1
      elif other.vertices[2][1] > self.y:
        line1 = LineString([(self.x, other.vertices[2][1]), other.vertices[2]])
        p = line1.intersection(circle)
        try: return other.vertices[2][0] - p.x
        except: return 1
      else: return other.bounds[0] - (self.x + self.prop)
    else:
      if other.vertices[0][1] < self.y:
        line1 = LineString([other.vertices[0], (self.x, other.vertices[0][0])])
        p = line1.intersection(circle)
        try: return p.x - other.vertices[0][0]
        except: return 1
      elif other.vertices[3][1] > self.y:
        line1 = LineString([other.vertices[3], (self.x, other.vertices[3][0])])
        p = line1.intersection(circle)
        try: return p.x - other.vertices[3][0]
        except: return 1
      else: return self.x - self.prop - other.bounds[2]

  def square_circle_distance(self, other, direction):
    circle = other.contours
    if direction == "down":
      if self.vertices[2][0] > other.x:
        line1 = LineString([(self.vertices[2][0], other.y), self.vertices[2]])
        p = line1.intersection(circle)
        try: return self.vertices[2][1] - p.y
        except: return 1
      elif self.vertices[3][0] < other.x:
        line1 = LineString([(self.vertices[3][0], other.y), self.vertices[3]])
        p = line1.intersection(circle)
        try: return self.vertices[3][1] - p.y
        except: return 1
      else: return self.bounds[1] - (other.y + other.prop)
    elif direction == "up":
      if self.vertices[1][0] > other.x:
        line1 = LineString([self.vertices[1], (self.vertices[1][0], other.y)])
        p = line1.intersection(circle)
        try: return p.y - self.vertices[1][1]
        except: return 1
      elif self.vertices[0][0] < other.x:
        line1 = LineString([self.vertices[0], (self.vertices[0][0], other.y)])
        p = line1.intersection(circle)
        try: return p.y - self.vertices[0][1]
        except: return 1
      else: return other.y - other.prop - self.bounds[3]
    elif direction == "right":
      if self.vertices[3][1] > other.y:
        line1 = LineString([self.vertices[3], (other.x, self.vertices[3][1])])
        p = line1.intersection(circle)
        try: return p.x - self.vertices[3][0]
        except: return 1
      elif self.vertices[0][1] < other.y:
        line1 = LineString([self.vertices[0], (other.x, self.vertices[0][1])])
        p = line1.intersection(circle)
        try: return p.x - self.vertices[0][0]
        except: return 1
      else: return other.x - other.prop - self.bounds[2]
    else:
      if self.vertices[2][1] > other.y:
        line1 = LineString([(other.x, self.vertices[2][1]), self.vertices[2]])
        p = line1.intersection(circle)
        try: return self.vertices[2][0] - p.x
        except: return 1
      elif self.vertices[1][1] < other.y:
        line1 = LineString([(other.x, self.vertices[1][1]), self.vertices[1]])
        p = line1.intersection(circle)
        try: return self.vertices[1][0] - p.x
        except: return 1
      else: return self.bounds[0] - (other.x + other.prop)

  def square_square_distance(self, other, direction):
    if direction == "down":
      return self.bounds[1] - other.bounds[3]
    elif direction == "up":
      return other.bounds[1] - self.bounds[3]
    elif direction == "right":
      return other.bounds[0] - self.bounds[2]
    else:
      return self.bounds[0] - other.bounds[2]

  
  
  
  def circle_triangle_distance(self, other, direction):
    r = self.prop
    if direction == "down":
      if not (self.x - r >= other.vertices[0][0] or self.x + r <= other.vertices[0][0]):
        return other.bounds[3] - (self.y - r)  + 1e-5
      else:
        if self.x <= other.vertices[0][0]:
          p = self.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
          line2 = LineString([other.vertices[0], other.vertices[1]])
        elif self.x > other.vertices[0][0]:
          p = self.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
          line2 = LineString([other.vertices[0], other.vertices[2]])
        line1 = LineString([p, (p[0],1)])
        point = line1.intersection(line2)
        try: return point.y - p[1]
        except: return 1         
    elif direction == "up":
      return (self.y + r) - other.bounds[1] + 1e-5
    elif direction == "right":
      if self.y - r <= other.vertices[0][1] and self.y - r >= other.vertices[1][1]:
        p = self.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
        line1 = LineString([(0, self.y), p]) 
        line2 = LineString([other.vertices[0], other.vertices[1]])
        point = line1.intersection(line2)
        try: return p[0] - point.x 
        except: return 1
      else:
        line1 = LineString([other.vertices[1], other.vertices[2]])
        line2 = LineString(self.vertices)
        p = line1.intersection(line2)
        try: return p.x - other.vertices[1][0]  + 1e-5
        except: return 1
    else:
      if self.y - r <= other.vertices[0][1] and self.y -r >= other.vertices[2][1]:
        p = self.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
        line1 = LineString([p, (1, self.y)])
        line2 = LineString([other.vertices[0], other.vertices[2]])
        point = line1.intersection(line2)
        try: return point.x - p[0]
        except: return 1
      else:
        line1 = LineString([other.vertices[1], other.vertices[2]])
        line2 = LineString(self.vertices)
        p = line1.intersection(line2)
        try: return other.vertices[2][0] - p.x  + 1e-5
        except: return 1

  def square_triangle_distance(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      if self_vert[3][0] < other_vert[0][0]:
        line1 = LineString([other_vert[0], other_vert[1]])
        line2 = LineString([(self_vert[3][0], 0), self_vert[3]])
        point = line1.intersection(line2)
        try: return self_vert[3][1] - point.y 
        except: return 1
      elif self_vert[2][0] > other_vert[0][0]:
        line1 = LineString([other_vert[0], other_vert[2]])
        line2 = LineString([(self_vert[2][0], 0), self_vert[2]])
        point = line1.intersection(line2)
        try: return self_vert[2][1] - point.y 
        except: return 1
      else:
        return self.bounds[1] - other.bounds[3]
    elif direction == "up":
      return other.bounds[1] - self.bounds[3]
    elif direction == "right":
      if self.bounds[1] >= other.bounds[1]: 
        line1 = LineString([other_vert[0], other_vert[1]])
        line2 = LineString([self_vert[3], (1, self_vert[3][1])])
        point = line1.intersection(line2)
        try: return point.x - self_vert[3][0] 
        except: return 1
      else:
        return other.bounds[0] - self.bounds[2]
    else:
      if self.bounds[1] >= other.bounds[1]:
        line1 = LineString([other_vert[0], other_vert[2]])
        line2 = LineString([(0, self_vert[2][1]), self_vert[2]])
        point = line1.intersection(line2)
        try: return self_vert[2][0] - point.x
        except: return 1
      else:
        return self.bounds[0] - other.bounds[2]

  def triangle_circle_distance(self, other, direction):
    r = other.prop
    if direction == "down": return self.bounds[1] - (other.y + r)
    elif direction == "up":
      if not (other.x + r <= self.vertices[0][0] or other.x - r >= self.vertices[0][0]):   
        return other.y - r - self.bounds[3]
      else:
        line1 = LineString([self.vertices[0], other.position])
        p1 = line1.intersection(other.polygon)
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: return p1[0][1] - self.vertices[0][1]
    elif direction == "right":
      if self.vertices[2][1] >= other.y + r * np.sin((5*np.pi)/4):
        p1 = other.polygon.intersection(LineString([self.vertices[2], other.position]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: return p1[0][0] - self.vertices[2][0]
      elif self.vertices[0][1] < other.y + r * np.sin((5*np.pi)/4):
        p1 = other.polygon.intersection(LineString([self.vertices[2], other.position]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: return p1[0][0] - self.vertices[0][0]
      else:
        p = other.position + (r * np.cos((5*np.pi)/4), r * np.sin((5*np.pi)/4))
        line1 = LineString([(0, p[1]), p])
        line2 = LineString([self.vertices[0], self.vertices[2]])
        point = line1.intersection(line2)
        try: return p[0] - point.x
        except: return 1
    else:
      if self.vertices[1][1] >= other.y + r * np.sin((7*np.pi)/4):
        p1 = other.polygon.intersection(LineString([other.position, self.vertices[1]]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: return self.vertices[1][0] - p1[0][0]
      elif self.vertices[0][1] < other.y + r * np.sin((7*np.pi)/4):
        p1 = other.polygon.intersection(LineString([other.position, self.vertices[0]]))
        p1 = [p for p in list(p1.coords) if Point(p) != Point(other.position)]
        if p1: return self.vertices[0][0] - p1[0][0]
      else:
        p = other.position + (r * np.cos((7*np.pi)/4), r * np.sin((7*np.pi)/4))
        line1 = LineString([p, (1, p[1])])
        line2 = LineString([self.vertices[0], self.vertices[1]])
        point = line1.intersection(line2)
        try: return point.x - p[0]
        except: return 1

  def triangle_square_distance(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      return self.bounds[1] - other.bounds[3]
    elif direction == "up":
      if other_vert[3][0] < self_vert[0][0]:
        line1 = LineString([self_vert[0], self_vert[1]])
        line2 = LineString([(other_vert[3][0], 0), other_vert[3]])
        point = line1.intersection(line2)
        try: return other_vert[3][1] - point.y 
        except: return 1
      elif other_vert[2][0] > self_vert[0][0]:
        line1 = LineString([self_vert[0], self_vert[2]])
        line2 = LineString([(other_vert[2][0], 0), other_vert[2]])
        point = line1.intersection(line2)
        try: return other_vert[2][1] - point.y 
        except: return 1
      else:
        return other.bounds[1] - self.bounds[3]
    elif direction == "right":
      if self.bounds[1] >= other.bounds[1]:
        return other.bounds[0] - self.bounds[2]
      else:
        line1 = LineString([self_vert[0], self_vert[2]])
        line2 = LineString([(0, other_vert[2][1]), other_vert[2]])
        point = line1.intersection(line2)
        try: return other_vert[2][0] - point.x 
        except: return 1
    else:
      if self.bounds[1] >= other.bounds[1]:
        return self.bounds[0] - other.bounds[2]
      else:
        line1 = LineString([self_vert[0], self_vert[1]])
        line2 = LineString([other_vert[3], (1, other_vert[3][1])])
        point = line1.intersection(line2)
        try: return point.x - other_vert[3][0] 
        except: return 1

  def triangle_triangle_distance(self, other, direction):
    self_vert = self.vertices
    other_vert = other.vertices
    if direction == "down":
      if self_vert[2][0] < other_vert[0][0]:
        line1 = LineString([(self_vert[2][0], 0), self_vert[2]])
        line2 = LineString([other_vert[0], other_vert[1]])
        point = line1.intersection(line2)
        try: return self_vert[2][1] - point.y
        except: return 1
      elif self_vert[1][0] > other_vert[0][0]:
        line1 = LineString([(self_vert[1][0], 0), self_vert[1]])
        line2 = LineString([other_vert[0], other_vert[2]])
        point = line1.intersection(line2)
        try: return self_vert[1][1] - point.y
        except: return 1
      else:
        return self.bounds[1] - other.bounds[3]
    elif direction == "up":
      if other_vert[2][0] < self_vert[0][0]:
        line1 = LineString([(other_vert[2][0], 0), other_vert[2]])
        line2 = LineString([self_vert[0], self_vert[1]])
        point = line1.intersection(line2)
        try: return other_vert[2][1] - point.y
        except: return 1
      elif other_vert[1][0] > self_vert[0][0]:
        line1 = LineString([(other_vert[1][0], 0), other_vert[1]])
        line2 = LineString([self_vert[0], self_vert[2]])
        point = line1.intersection(line2)
        try: return other_vert[1][1] - point.y
        except: return 1
      else:
        return other.bounds[1] - self.bounds[3]
    elif direction == "right":
      if self.bounds[1] > other.bounds[1]:
        line1 = LineString([self_vert[2], (1, self_vert[2][1])])
        line2 = LineString([other_vert[0], other_vert[1]])
        point = line1.intersection(line2)
        try: return point.x - self_vert[2][0]
        except: return 1
      elif self.bounds[1] < other.bounds[1]:
        line1 = LineString([(0, other_vert[1][1]), other_vert[1]])
        line2 = LineString([self_vert[0], self_vert[2]])
        point = line1.intersection(line2)
        try: return other_vert[1][0] - point.x
        except: return 1
      else:
        return other.bounds[0] - self.bounds[2]
    else:
      if self.bounds[1] > other.bounds[1]:
        line1 = LineString([(0, self_vert[1][1]), self_vert[1]])
        line2 = LineString([other_vert[0], other_vert[2]])
        point = line1.intersection(line2)
        try: return self_vert[1][0] - point.x
        except: return 1
      elif self.bounds[1] < other.bounds[1]:
        line1 = LineString([other_vert[2], (1, other_vert[2][1])])
        line2 = LineString([self_vert[0], self_vert[1]])
        point = line1.intersection(line2)
        try: return point.x - other_vert[2][0]
        except: return 1
      else:
        return other.bounds[2] - self.bounds[0]
  

  @property
  def vertices(self):
    """Numpy array of vertices of the shape."""
    transform = mpl_transforms.Affine2D().translate(*self._position)
    path = transform.transform_path(self._centered_path)
    return path.vertices

  @property
  def out_of_frame(self):
    return not (np.all(self._position >= [0., 0.]) and np.all(self._position <= [1., 1.]))

  @property
  def x(self):
    return self._position[0]

  @property
  def y(self):
    return self._position[1]

  @property
  def shape(self):
    return self._shape

  @shape.setter
  def shape(self, s):
    self._shape = s
    self._reset_centered_path()

  @property
  def angle(self):
    return self._angle

  @angle.setter
  def angle(self, a):
    rotate = mpl_transforms.Affine2D().rotate_deg(a - self._angle)
    self._centered_path = rotate.transform_path(self._centered_path)
    self._angle = a

  @property
  def scale(self):
    return self._scale

  @scale.setter
  def scale(self, s):
    rescale = mpl_transforms.Affine2D().scale(s - self._scale)
    self._centered_path = rescale.transform_path(self._centered_path)
    self._scale = s

  @property
  def c0(self):
    return self._color[0]

  @property
  def c1(self):
    return self._color[1]

  @property
  def c2(self):
    return self._color[2]

  @property
  def x_vel(self):
    return self._velocity[0]

  @property
  def y_vel(self):
    return self._velocity[1]

  @property
  def color(self):
    return self._color

  @property
  def position(self):
    return self._position

  @property
  def velocity(self):
    return self._velocity

  @property
  def factors(self):
    factors = collections.OrderedDict()
    for factor_name in FACTOR_NAMES:
      factors[factor_name] = getattr(self, factor_name)
    return factors

  @property
  def polygon(self):
    return Polygon(self.vertices)

  @property
  def bounds(self):
    return self.polygon.bounds

  @property  
  def offsets(self): 
    bottom_left = np.abs((self.bounds[0], self.bounds[1]) - self._position) # (min_x, min_y)
    top_right = np.abs((self.bounds[2], self.bounds[3]) - self._position) # (max_x, max_y)
    return bottom_left, top_right

  @property
  def contours(self):
    points = list(self.vertices)
    points.append(points[0])
    return LineString(points) 
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Spriteworld sprite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from spriteworld.abstractsprite import AbstractSprite, FACTOR_NAMES
from shapely.geometry import LineString, Point

from spriteworld.utils import *

# Just to catch infinite while-looping. Anything >1e4 should be plenty safe.
_MAX_TRIES = int(1e6)
class Sprite(AbstractSprite):
  """Sprite class.

  Sprites are simple shapes parameterized by a few factors (position, shape,
  angle, scale, color, velocity). They are the building blocks of Spriteworld,
  so every Spriteworld environment state is simple a collection of sprites.

  We assume that (x, y) are in mathematical coordinates, i.e. (0, 0) is at the
  lower-left of the frame.
  """

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
    super().__init__(x, y, shape, angle, scale, c0, c1, c2, x_vel, y_vel)

    self._geometrical_properties()


  def _geometrical_properties(self):
    # circle: area = r**2 * pi -> r = sqrt(area/pi)
    # square: area = s**2 -> s = sqrt(area)
    area = self.polygon.area
    if self.shape == "circle":
      self.prop = np.sqrt(area / np.pi)
    if self.shape == "square":
      self.prop = np.sqrt(area) 

  def handle_collision(self, other, direction):
    if self.shape == "circle" and other.shape == "circle":
      self._handle_circle_circle(other, direction)
    elif self.shape == "circle" and other.shape == "square": 
      self._handle_circle_square(other, direction)
    elif self.shape == "circle" and other.shape == "triangle":
      self._handle_circle_triangle(other, direction)
    elif self.shape == "square" and other.shape == "circle":
      self._handle_square_circle(other, direction)
    elif self.shape == "square" and other.shape == "square":
      self._handle_square_square(other, direction)
    elif self.shape == "square" and other.shape == "triangle": 
      self._handle_square_triangle(other, direction)
    elif self.shape == "triangle" and other.shape == "circle":
      self._handle_triangle_circle(other, direction)
    elif self.shape == "triangle" and other.shape == "square":
      self._handle_triangle_square(other, direction)
    elif self.shape == "triangle" and other.shape == "triangle": 
      self._handle_triangle_triangle(other, direction)
    else:
      exit("Unexpected shapes")

  def resolve_overlapping(self, other, direction):
    if self.shape == "circle" and other.shape == "circle":
      self._circle_circle_overlap(other, direction)
    elif self.shape == "circle" and other.shape == "square": 
      self._circle_square_overlap(other, direction)
    elif self.shape == "circle" and other.shape == "triangle": 
      self._circle_triangle_overlap(other, direction)
    elif self.shape == "square" and other.shape == "circle":
      self._square_circle_overlap(other, direction)
    elif self.shape == "square" and other.shape == "square":
      self._square_square_overlap(other, direction)
    elif self.shape == "square" and other.shape == "triangle": 
      self._square_triangle_overlap(other, direction)
    elif self.shape == "triangle" and other.shape == "circle":
      self._triangle_circle_overlap(other, direction)
    elif self.shape == "triangle" and other.shape == "square":
      self._triangle_square_overlap(other, direction)
    elif self.shape == "triangle" and other.shape == "triangle": 
      self._triangle_triangle_overlap(other, direction)
    else:
      exit("Unexpected shape")

  def avoid_overlapping(self, other, direction, resolve=True):
    if self.shape == "circle":
      if other.shape == "circle":
        overlapping = circle_circle(self.position, other.position, self.prop, other.prop)
      elif other.shape == "square":
        overlapping = square_circle(self.position, other.position, self.prop, other.prop)
      elif other.shape == "triangle":
        overlapping = triangle_circle(other.vertices, self.position, self.prop)
    
    elif self.shape == "square":
      if other.shape == "circle":
        overlapping = square_circle(other.position, self.position, other.prop, self.prop)
      elif other.shape == "square":
        overlapping = self.check_collision(other)
      elif other.shape == "triangle":
        overlapping = triangle_square(other.vertices, self.bounds)
        
    elif self.shape == "triangle":
      if other.shape == "circle":
        overlapping = triangle_circle(self.vertices, other.position, other.prop) 
      elif other.shape == "square":
        overlapping = triangle_square(self.vertices, other.bounds)
      elif other.shape == "triangle":
        overlapping = TriTri2D(self.vertices, other.vertices, allowReversed=True)

    if overlapping and resolve: 
      self.resolve_overlapping(other, direction)
    if not resolve: return overlapping

  def detect_collision(self, other, motion, direction):
    if self.shape == "circle":    
      if other.shape == "circle":
        collision = circle_circle(self.position + motion, other.position, self.prop, other.prop)
      elif other.shape == "square":
        collision = square_circle(self.position + motion, other.position, self.prop, other.prop)
      elif other.shape == "triangle":
        collision = triangle_circle(other.vertices, self._position + motion, self.prop)

    elif self.shape == "square":
      if other.shape == "circle":
        collision = square_circle(other.position, self._position + motion, other.prop, self.prop)
      elif other.shape == "square":
        self._position += motion
        collision = self.check_collision(other)
        self._position -= motion
      elif other.shape == "triangle":
        self._position += motion
        collision = triangle_square(other.vertices, self.bounds)
        self._position -= motion
  
    elif self.shape == "triangle":
      p1 = self.vertices[1][1]
      p2 = self.vertices[2][1]  
      self._position += motion
      if other.shape == "circle":
        collision = triangle_circle(self.vertices, other._position, other.prop)
      elif other.shape == "square":
        collision = triangle_square(self.vertices, other.bounds)

      elif other.shape == "triangle":
        p3 = self.vertices[1][1]
        p4 = self.vertices[2][1]
        collision = TriTri2D(self.vertices, other.vertices, allowReversed=True)
        condition2 = False
        if direction == "down":
          if self.vertices[2][0] < other.vertices[0][0] and self.vertices[2][0] >= other.vertices[1][0]:
            condition2 = other.vertices[1][1] <= p1 and other.vertices[1][1] >= p3
          if self.vertices[1][0] > other.vertices[0][0] and self.vertices[1][0] <= other.vertices[2][0]:
            condition2 = other.vertices[2][1] <= p2 and other.vertices[2][1] >= p4
        if direction == "up":
          if self.vertices[2][0] < other.vertices[0][0] and self.vertices[2][0] >= other.vertices[1][0]:
            condition2 = other.vertices[1][1] <= p3 and other.vertices[1][1] >= p1
          if self.vertices[1][0] > other.vertices[0][0] and self.vertices[1][0] <= other.vertices[2][0]:
            condition2 = other.vertices[2][1] <= p4 and other.vertices[2][1] >= p2
        collision = collision or condition2
      self._position -= motion

    #if collision: self.handle_collision(other, direction)     
    return collision

  def get_carried_sprite(self, sprites, motion, direction):
    """sprites doesn't contain this and agent sprite"""
    return [s for s in sprites if self.detect_collision(s, motion, direction)]
    
  def move(self, motion, keep_in_frame=False, others=None):
    """Move the sprite, optionally keeping its centerpoint within the frame."""

    if others is not None:
      direction = get_motion_direction(motion)
      carried_sprites = self.get_carried_sprite(others, motion, direction)
      if carried_sprites:
        cs, carried_sprites = get_closest(self, carried_sprites, direction)
        self.handle_collision(cs, direction)
        positions = get_relative_positions(self, cs)
        print(positions)
        if direction in [opposite[p] for p in positions]:
          self._position += motion
          others_ = [sprite for sprite in others if sprite != cs]
          cs.move(motion, keep_in_frame, others_)
          self.avoid_overlapping(cs, direction)
        else:
          self._position += motion
        for carried_sprite in carried_sprites:
          if self.avoid_overlapping(carried_sprite, direction, False):
            others_ = [sprite for sprite in others if sprite != carried_sprite]
            carried_sprite.move(motion, keep_in_frame, others_)
            self.avoid_overlapping(carried_sprite, direction)           
      else: 
          self._position += motion            
    else: self._position += motion

    if keep_in_frame:
      bottom_left, top_right = self.offsets
      self._position = np.clip(self._position, 0.0 + bottom_left, 1.0 - top_right)

  def update_position(self, keep_in_frame=False, others=None):
    """Update position based on velocity."""
    self.move(self.velocity, keep_in_frame=keep_in_frame, others=others)

  def contains_point(self, point):
    """Check if the point is contained in the Sprite."""
    return self._centered_path.contains_point(point - self.position)

  def check_collision(self, other):
    """Check if this sprite collides with the other"""
    return self.polygon.intersects(other.polygon)

  def sample_contained_position(self):
    """Sample random position uniformly within sprite."""
    low = np.min(self._centered_path.vertices, axis=0)
    high = np.max(self._centered_path.vertices, axis=0)
    for _ in range(_MAX_TRIES):
      sample = self._position + np.random.uniform(low, high)
      if self.contains_point(sample):
        return sample
    raise ValueError('max_tries exceeded. There is almost surely an error in '
                     'the SpriteWorld library code.')

      

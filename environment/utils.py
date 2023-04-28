import numpy as np
from shapely.geometry import LineString

# Defining region codes
INSIDE = 0 # 0000
LEFT = 1 # 0001
RIGHT = 2 # 0010
BOTTOM = 4 # 0100
TOP = 8	 # 1000

opposite = {"right": "left", "left": "right", "up": "down", "down": "up"}

# Function to compute region code for a point(x, y)
def computeCode(x, y, bounds):
	x_min, y_min, x_max, y_max = bounds
	code = INSIDE
	if x < x_min:	 # to the left of rectangle
		code |= LEFT
	elif x > x_max: # to the right of rectangle
		code |= RIGHT
	if y < y_min:	 # below the rectangle
		code |= BOTTOM
	elif y > y_max: # above the rectangle
		code |= TOP

	return code

# Clipping a line from P1 = (x1, y1) to P2 = (x2, y2)
def cohenSutherlandClip(x1, y1, x2, y2, bounds):

	x_min, y_min, x_max, y_max = bounds
	# Compute region codes for P1, P2
	code1 = computeCode(x1, y1, bounds)
	code2 = computeCode(x2, y2, bounds)
	accept = False

	while True:

		# If both endpoints lie within rectangle
		if code1 == 0 and code2 == 0:
			accept = True
			break

		# If both endpoints are outside rectangle
		elif (code1 & code2) != 0:
			break

		# Some segment lies within the rectangle
		else:

			# Line Needs clipping
			# At least one of the points is outside,
			# select it
			x = 1.0
			y = 1.0
			if code1 != 0:
				code_out = code1
			else:
				code_out = code2

			# Find intersection point
			# using formulas y = y1 + slope * (x - x1),
			# x = x1 + (1 / slope) * (y - y1)
			if code_out & TOP:
			
				# point is above the clip rectangle
				x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
				y = y_max

			elif code_out & BOTTOM:
				
				# point is below the clip rectangle
				x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
				y = y_min

			elif code_out & RIGHT:
				
				# point is to the right of the clip rectangle
				y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
				x = x_max

			elif code_out & LEFT:
				
				# point is to the left of the clip rectangle
				y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
				x = x_min

			# Now intersection point x, y is found
			# We replace point outside clipping rectangle
			# by intersection point
			if code_out == code1:
				x1 = x
				y1 = y
				code1 = computeCode(x1, y1, bounds)

			else:
				x2 = x
				y2 = y
				code2 = computeCode(x2, y2, bounds)

	return accept

def CheckTriWinding(tri, allowReversed):
	trisq = np.ones((3,3))
	trisq[:,0:2] = np.array(tri)
	detTri = np.linalg.det(trisq)
	if detTri < 0.0:
		if allowReversed:
			a = trisq[2,:].copy()
			trisq[2,:] = trisq[1,:]
			trisq[1,:] = a
		else: raise ValueError("triangle has wrong winding direction")
	return trisq

def TriTri2D(t1, t2, allowReversed = False):
	#Trangles must be expressed anti-clockwise
	t1s = CheckTriWinding(t1, allowReversed)
	t2s = CheckTriWinding(t2, allowReversed)

	chkEdge = lambda x: np.linalg.det(x) < 0.0

	#For edge E of trangle 1,
	for i in range(3):
		edge = np.roll(t1s, i, axis=0)[:2,:]

		#Check all points of trangle 2 lay on the external side of the edge E. If
		#they do, the triangles do not collide.
		if (chkEdge(np.vstack((edge, t2s[0]))) and
			chkEdge(np.vstack((edge, t2s[1]))) and  
			chkEdge(np.vstack((edge, t2s[2])))):
			return False

	#For edge E of trangle 2,
	for i in range(3):
		edge = np.roll(t2s, i, axis=0)[:2,:]

		#Check all points of trangle 1 lay on the external side of the edge E. If
		#they do, the triangles do not collide.
		if (chkEdge(np.vstack((edge, t1s[0]))) and
			chkEdge(np.vstack((edge, t1s[1]))) and  
			chkEdge(np.vstack((edge, t1s[2])))):
			return False

	#The triangles collide
	return True

def circle_circle(pos1, pos2, rad1, rad2):
	return np.linalg.norm(pos1 - pos2) <= rad1 + rad2

def square_circle(cpos, spos, rad, side):
	dist = np.abs(cpos - spos)
	if dist[0] > (side/2 + rad): return False
	if dist[1] > (side/2 + rad): return False
	if dist[0] <= side/2 or dist[1] <= side/2: return True
	return np.linalg.norm(dist - side/2) <= rad

def triangle_circle(tvert, cpos, rad):
	# vertex within circle
	if np.linalg.norm(tvert[0] - cpos) <= rad: return True 
	if np.linalg.norm(tvert[1] - cpos) <= rad: return True
	if np.linalg.norm(tvert[2] - cpos) <= rad: return True

	# circle center within triangle
	a = (tvert[1] - tvert[0]) * (cpos - tvert[0])
	b = (tvert[2] - tvert[1]) * (cpos - tvert[1])
	c = (tvert[0] - tvert[2]) * (cpos - tvert[2])
	if a[0] >= a[1] and b[0] >= b[1] and c[0] >= c[1]: return True
	
	# circle intersects edge
	cvert = cpos - tvert
	edges = np.array([tvert[1] - tvert[0], tvert[2] - tvert[1], tvert[0] - tvert[2]])
	k = (cvert * edges).sum(axis=1)
	lens = np.sqrt((edges**2).sum(axis=1))
	k_positive = np.where(k > 0)[0]
	
	if len(k_positive) == 0: return False
	
	k = k[k_positive]
	lens = lens[k_positive]
	cvert = cvert[k_positive]
	half_k_higherthan_len = np.where(k/lens < lens)[0]
	
	if len(half_k_higherthan_len) == 0: return False

	k = (k[half_k_higherthan_len] / lens[half_k_higherthan_len])**2
	cvert = cvert[half_k_higherthan_len]
	k = np.sqrt((cvert**2).sum(axis=1) - k)

	return any(k <= rad) 


def triangle_square(tvert, bounds):
	coll = cohenSutherlandClip(tvert[0][0], tvert[0][1], tvert[1][0], tvert[1][1], bounds)
	if coll: return coll
	coll = cohenSutherlandClip(tvert[0][0], tvert[0][1], tvert[2][0], tvert[2][1], bounds)
	if coll: return coll
	return cohenSutherlandClip(tvert[1][0], tvert[1][1], tvert[2][0], tvert[2][1], bounds)
	

def get_motion_direction(motion):
	if motion[0] == 0 and motion[1] > 0:
		return "up"
	elif motion[1] == 0 and motion[0] > 0:
		return "right"
	elif motion[0] == 0 and motion[1] < 0:
		return "down"
	elif motion[1] == 0 and motion[0] < 0:
		return "left"
	else:
		exit("unexpected direction")

def get_relative_positions(s1, s2):
	# returns the position of s1 relative to s2
	if s1.shape == "circle":
		if s2.shape == "circle":
			directions = circle_circle_position(s1,s2) 
		elif s2.shape == "square":
			directions = circle_square_position(s1,s2)
		else:
			directions = circle_triangle_position(s1,s2)
	elif s1.shape == "square":
		if s2.shape == "circle":
			directions = circle_square_position(s2,s1, True) 
		elif s2.shape == "square":
			directions = square_square_position(s1,s2)
		else:
			directions = square_triangle_position(s1,s2)
	else:
		if s2.shape == "circle":
			directions = triangle_circle_position(s2,s1) 
		elif s2.shape == "square":
			directions = triangle_square_position(s1,s2)
		else:
			directions = triangle_triangle_position(s1,s2)

	return directions

def circle_circle_position(s1, s2):
	directions = []
	if s1.position[0] < s2.position[0]: directions.append("left")
	if s1.position[0] > s2.position[0]: directions.append("right")
	if s1.position[1] < s2.position[1]: directions.append("down")
	if s1.position[1] > s2.position[1]: directions.append("up")
	return directions

def circle_square_position(s1, s2, reverse=False):
	directions = []
	if s1.position[0] < s2.bounds[0]: directions.append("left")
	if s1.position[0] > s2.bounds[2]: directions.append("right")
	if s1.position[1] < s2.bounds[1]: directions.append("down")
	if s1.position[1] > s2.bounds[3]: directions.append("up")
	return [opposite[d] for d in directions] if reverse else directions

def circle_triangle_position(c, t):
	directions = []
	y = c.y + c.prop * np.sin((5*np.pi)/4)
	if c.x < t.x and c.bounds[1] < t.bounds[3] and c.bounds[3] > t.bounds[1]: directions.append("left")
	if c.x > t.x and c.bounds[1] < t.bounds[3] and c.bounds[3] > t.bounds[1]: directions.append("right")
	if y > t.bounds[1]: directions.append("up")
	if c.y + c.prop <= t.bounds[1] + 0.02: directions.append("down")
	return directions

def triangle_circle_position(c, t):
	directions = []
	if t.x < c.x and ((t.bounds[1] <= c.bounds[3] and t.bounds[3] >= c.bounds[1]) or (t.bounds[3] <= c.bounds[3] and t.bounds[1] >= c.bounds[1])): directions.append("left")
	if t.x > c.x and ((t.bounds[1] <= c.bounds[3] and t.bounds[3] >= c.bounds[1]) or (t.bounds[3] <= c.bounds[3] and t.bounds[1] >= c.bounds[1])): directions.append("right")
	if t.bounds[1] > c.y: directions.append("up")
	if t.bounds[1] <= c.y: directions.append("down")
	return directions

def square_square_position(s1, s2):
	directions = []
	if s1.position[0] <= s2.bounds[0]: directions.append("left")
	if s1.position[0] >= s2.bounds[2]: directions.append("right")
	if s1.position[1] <= s2.bounds[1]: directions.append("down")
	if s1.position[1] >= s2.bounds[3]: directions.append("up")
	return directions

def square_triangle_position(s, t):
	directions = []
	if s.x < t.x and s.bounds[1] < t.bounds[3] and s.bounds[3] > t.bounds[1]: directions.append("left")
	if s.x > t.x and s.bounds[1] < t.bounds[3] and s.bounds[3] > t.bounds[1]: directions.append("right")
	if s.bounds[1] > t.bounds[1]: directions.append("up")
	if s.y < t.bounds[1] and s.bounds[0] <= t.bounds[2] and s.bounds[2] >= t.bounds[0]: directions.append("down")
	return directions

def triangle_square_position(t, s):
	directions = []
	if t.x < s.x and t.bounds[1] <= s.bounds[3] and t.bounds[3] >= s.bounds[1]: directions.append("left")
	if t.x > s.x and t.bounds[1] <= s.bounds[3] and t.bounds[3] >= s.bounds[1]: directions.append("right")
	if t.y >= s.bounds[3]: directions.append("up")
	if t.bounds[1] < s.bounds[1] and t.bounds[2] >= s.bounds[0] and t.bounds[0] <= t.bounds[2]: directions.append("down")
	return directions

def triangle_triangle_position(t1, t2):
	directions = []
	if t1.x < t2.x and t1.bounds[1] < t2.bounds[3] and t1.bounds[3] > t2.bounds[1]: directions.append("left")
	if t1.x > t2.x and t1.bounds[1] < t2.bounds[3] and t1.bounds[3] > t2.bounds[1]: directions.append("right")
	if t1.bounds[1] > t2.bounds[1]: directions.append("up")
	if t1.y < t2.bounds[1] and t1.bounds[0] <= t2.bounds[2] and t1.bounds[2] >= t2.bounds[0]: directions.append("down")
	return directions

def _get_distance(s1, s2, direction):
	if s1.shape == "circle" and s2.shape == "circle":
		return s1.circle_circle_distance(s2, direction)
	elif s1.shape == "circle" and s2.shape == "square": 
		return s1.circle_square_distance(s2, direction)
	elif s1.shape == "circle" and s2.shape == "triangle": 
		return s1.circle_triangle_distance(s2, direction)
	elif s1.shape == "square" and s2.shape == "circle":
		return s1.square_circle_distance(s2, direction)
	elif s1.shape == "square" and s2.shape == "square":
		return s1.square_square_distance(s2, direction)
	elif s1.shape == "square" and s2.shape == "triangle": 
		return s1.square_triangle_distance(s2, direction)
	elif s1.shape == "triangle" and s2.shape == "circle":
		return s1.triangle_circle_distance(s2, direction)
	elif s1.shape == "triangle" and s2.shape == "square":
		return s1.triangle_square_distance(s2, direction)
	elif s1.shape == "triangle" and s2.shape == "triangle": 
		return s1.triangle_triangle_distance(s2, direction)
	else:
		exit("Unexpected shape") 
	
def get_closest(s, carried, direction):
	i = np.argmin([_get_distance(s, cs, direction) for cs in carried]) 
	closest = carried[i]
	carried.pop(i)
	return closest, carried

def direction_bound(s, s2, direction):
	if direction == "down": return abs(s.bounds[1] - s2.y)
	elif direction == "up": return abs(s.bounds[3] - s2.y)
	elif direction == "right": return abs(s.bounds[2] - s2.x)
	else: return abs(s.bounds[0]  - s2.x)


from abc import ABCMeta , abstractmethod
from math import cos , sin
import taichi as ti

@ti.data_oriented
class Ray:
    def __init__(self , origin  , direction ):
        self.origin = origin
        self.direction = direction

    def point_at( self , t : ti.f32):
        return origin + direction * t
        
@ti.data_oriented
class Transform:
    def __init__ (self ):
        pass

    @property
    def translation (self):
        return self._translation

    @translation.setter
    def translation(self , translation ):
        self._translation = translation

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self , orientation : float):
        self._orientation = orientation
        self._cos_angle = cos(orientation)
        self._sin_angle = sin(orientation)

    def to_local(self , point_in_world):
        return self.to_local_direction(point_in_world - self.translation)
    
    def to_local_direction(self , dir_in_world):
        return ti.Vector([
            self._cos_angle * dir_in_world[0] + self._sin_angle * dir_in_world[1] ,
            -self._sin_angle * dir_in_world[0] + self._cos_angle * dir_in_world[1]
        ])

    def to_world(self , point_in_local):
        return  self.to_world_direction(point_in_local) + self.translation

    def to_world_direction(self , dir_in_local):
        return ti.Vector([
            self._cos_angle * point_in_local[0] - self._sin_angle * point_in_local[1] + self.translation[0] , 
            self._sin_angle * point_in_local[0] - self._cos_angle * point_in_local[1] + self.translation[1]
        ])

@ti.data_oriented
class BoundingBox:
    def __init__(self , point1 , point2 ):
        self.lower_corner = [min(point1[0] ,point2[0]) , min(point1[1] , point2[1])]
        self.upper_corner = [max(point1[0] ,point2[0]) , max(point2[1] , point2[1])]

    @property
    def width(self):
        return self.upper_corner[0] - self.lower_corner[0]

    @property
    def height(self):
        return self.upper_corner[1] - self.lower_corner[1]

    def overlaps(self , other) -> bool:
        return not ( \
            (self.upper_corner[0] < other.upper_corner[0] or self.lower_corner[0] > other.lower_corner[0]) \
            or
            (self.upper_corner[1] < other.upper_corner[1] or self.lower_corner[1] > other.upper_corner[1]) \
        )

    def contains(self , point)->bool :
        return self.upper_corner[0] >= point[0] and self.lower_corner[0] <= point[0] \
            and self.upper_corner[1] >= point[1] and self.lower_corner[1] <= point[1]           

@ti.data_oriented
class Surface(metaclass = ABCMeta):

    def __init__(self, transfrom , is_normal_flipped):
        self.transform  = transfrom
        self.is_normal_flipped = is_normal_flipped

    def closestpoint(self , other_point):
        pass

    @abstractmethod
    def closestnormal(self):
        pass

    @abstractmethod
    def boundingBox(self):
        pass

    @abstractmethod
    def intersects(self  , ray : Ray)->bool:
        

    @abstractmethod
    def closestDistance(self , point )->float:
        pass

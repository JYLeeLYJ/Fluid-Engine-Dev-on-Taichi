from abc import ABCMeta , abstractmethod
from math import cos , sin
import taichi as ti

from helper_functions import clamp , distance , distance_sqr
from basic_types import Vector , Float , Int

"""
TODO: Not well complete . 
    1. bounding_box query 
    2. ray intersects check

"""

@ti.data_oriented
class Ray:
    def __init__(self , origin : Vector , direction :Vector):
        self.origin = origin
        self.direction = direction

    @ti.func
    def point_at( self , t : Float ):
        return self.origin + self.direction * t

@ti.data_oriented
class BoundingBox:
    def __init__(self , point1 , point2 ):
    # def __init__(self , lower_corner , upper_corner):
        self.lower_corner = ti.Vector([min(point1[0] ,point2[0]) , min(point1[1] , point2[1])])
        self.upper_corner = ti.Vector([max(point1[0] ,point2[0]) , max(point2[1] , point2[1])])

    @ti.func
    def width(self):
        return self.upper_corner[0] - self.lower_corner[0]

    @ti.func
    def height(self):
        return self.upper_corner[1] - self.lower_corner[1]

    """
    # def overlaps(self , other) -> bool:
    #     return not ( \
    #         (self.upper_corner[0] < other.upper_corner[0] or self.lower_corner[0] > other.lower_corner[0]) \
    #         or
    #         (self.upper_corner[1] < other.upper_corner[1] or self.lower_corner[1] > other.upper_corner[1]) \
    #     )
    """

    @ti.func
    def contains(self , point)->bool :
        return self.upper_corner[0] >= point[0] and self.lower_corner[0] <= point[0] \
            and self.upper_corner[1] >= point[1] and self.lower_corner[1] <= point[1]      


    """
    # def intersects(self , ray)->bool:
    #     tmin , tmax = 0 , 1e8
    #     ray_dir_inv = ti.Vector([1.0/ray.direction[0] , 1.0/ray.direction[1]])

    #     ret = True

    #     for i in ti.static(range(2)):
    #         near = (self.lower_corner[i] - ray.origin[i]) * ray_dir_inv[i]
    #         far = (self.upper_corner[i] - ray.origin[i]) * ray_dir_inv[i]

    #         if near > far :
    #             near ,far = far , near
            
    #         tmin = max(near , tmin)
    #         tmax = min(far , tmax)

    #         if tmin > tmax :
    #             ret = False

    #     return ret
    """

@ti.data_oriented
class Transform:
    def __init__ (self ):
        self.translation = ti.Vector([0.0 ,0.0])
        self.orientation = 0.0      


    @property
    def translation (self):
        return self._translation

    @translation.setter
    def translation(self , translation : Vector ):
        self._translation = translation

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self , orientation : float):
        self._orientation = orientation
        self._cos_angle = cos(orientation)
        self._sin_angle = sin(orientation)

    @ti.func
    def to_local(self , point_in_world : Vector) -> Vector:
        return self.to_local_direction(point_in_world - self.translation)
    
    @ti.func
    def to_local_direction(self , dir_in_world : Vector) -> Vector:
        #TODO use rotation matrix
        return ti.Vector([
            self._cos_angle * dir_in_world[0] + self._sin_angle * dir_in_world[1] ,
            -self._sin_angle * dir_in_world[0] + self._cos_angle * dir_in_world[1]
        ])

    @ti.func
    def to_local_ray(self , ray : Ray) -> Ray:
        return  Ray(self.to_local(ray.origin) , self.to_local_direction(ray.direction))

    """
    # def to_local_boundingbox(self , box : BoundingBox) -> BoundingBox:
    #     pass
    """

    @ti.func
    def to_world(self , point_in_local : Vector)->Vector:
        return  self.to_world_direction(point_in_local) + self.translation

    @ti.func
    def to_world_direction(self , dir_in_local : Vector)->Vector:
        return ti.Vector([
            self._cos_angle * dir_in_local[0] - self._sin_angle * dir_in_local[1] + self.translation[0] , 
            self._sin_angle * dir_in_local[0] - self._cos_angle * dir_in_local[1] + self.translation[1]
        ])

    @ti.func
    def to_world_ray(self, ray : Ray) -> Ray:
        return Ray(self.to_world(ray.origin) , self.to_world_direction(ray.direction))

    """
    # def to_world_boundingbox(self , box : BoundingBox) -> BoundingBox:
    #     pass
    """

"""
# class SurfaceRayIntersection:
#     def __init__(self):
#         self.is_intersecting = False
#         self.distance = 1e8
#         self.point = [0.0 ,0.0]
#         self.normal = [0.0 ,0.0]

#     def to_world(self , transform : Transform , is_normal_flipped : bool) :
#         self.point = transform.to_world(self.point)
#         self.normal = transform.to_world_direction(self.normal) 

#         if ti.static(is_normal_flipped is True):
#             self.normal *= -1,0
"""

@ti.data_oriented
class Surface(metaclass = ABCMeta):

    def __init__(self, transfrom : Transform, is_normal_flipped : bool):
        self.transform  = transfrom
        self.is_normal_flipped = is_normal_flipped

    @ti.func
    def closest_point(self , point : Vector) -> Vector:
        return self.transform.to_world(self.closest_point_local(self.transform.to_local(point)))

    @ti.func
    def closest_normal(self , point : Vector ) -> Vector :
        res = self.transform.to_world_direction(self.closest_normal_local(point))
        if ti.static(self.is_normal_flipped):
            res *= -1.0
        return res

    """
    # def bounding_box(self):
    #     pass

    # def intersects(self  , ray : Ray) ->bool:
    #     return self.intersects_local(self.transform.to_local_ray(ray))

    # def closest_intersection(self , ray : Ray) :
    #     res = self.closest_interesection_local(self.transform.to_local_ray(ray))
    #     res.to_world(self.transform , self.is_normal_flipped)
    #     return res
    """
    
    @ti.func
    def closest_distance(self , point : Vector)->float:
        return self.closest_distance_local(self.transform.to_local(point))

    @ti.func
    def is_inside(self , point : Vector) -> bool :
        return self.is_normal_flipped is not self.is_inside_local(self.transform.to_local(point))

    ### ====== abstract methods =================================
    # computations in local space

    @abstractmethod
    def closest_point_local(self , point) -> Vector:
        pass

    """
    # @abstractmethod
    # def bounding_box_local(self)-> BoundingBox:
    #     pass

    # @abstractmethod
    # def closest_interesection_local(self , ray : Ray):
    #     pass
    """
    
    @abstractmethod
    def closest_normal_local(self , point) ->Vector:
        pass

    """
    # def intersects_local(self , ray : Ray) ->bool:
    #     return self.closest_interesection_local(ray).is_intersecting
    """

    @ti.func
    def closest_distance_local(self , point:Vector) -> Float:
        return distance(point , self.closest_point_local(point))

    @ti.func
    def is_inside_local(self , point : Vector )->bool :
        pl = self.closest_point_local(point)
        nl = self.closest_normal_local(point)
        return (point - pl).dot(nl) < 0.0
    

class ImplicitSurface(Surface):
    def __init__(self , transform : Transform , is_normal_flipped : bool):
        super().__init__(transform, is_normal_flipped)

    @abstractmethod
    def sign_distance_local(self ,point : Vector) -> Float:
        pass

    @ti.func
    def sign_distance(self , point : Vector) -> Float :
        dis = self.sign_distance_local(self.transform.to_local(point))
        return -dis if self.is_normal_flipped else dis

    @ti.func
    def closest_distance_local(self, point :Vector)->Float:
        return abs(self.sign_distance_local(point))

    @ti.func
    def is_inside_local(self ,point: Vector)->bool:
        return self.sign_distance_local(point) < 0.0

class SurfaceToImplict(ImplicitSurface):
    def __init__(
        self , 
        surface : Surface , 
        transform : Transform = Transform() , 
        is_normal_flipped : bool = False):

        super().__init__(transform , is_normal_flipped)
        self._surface = surface

    def surface(self) -> Surface :
        return self._surface

    @ti.func
    def closest_point_local(self , point : Vector)->Vector:
        return self._surface.closest_point_local(point)

    @ti.func
    def closest_distance_local(self , point : Vector)->Float:
        return self._surface.closest_distance_local(point)

    """
    # def intersects_local(self , ray  : Ray) -> bool :
    #     return self._surface.intersects(ray)

    # def bounding_box_local(self) -> BoundingBox:
    #     return self._surface.bounding_box()
    """

    @ti.func
    def closest_normal_local(self , point : Vector) -> Vector:
        return self._surface.closest_normal_local(point)

    @ti.func
    def sign_distance_local(self , point : Vector) -> Float:
        p = self._surface.closest_point_local(point)
        return - distance(p , point) if self._surface.is_inside_local(point) else distance(p , point)

    """
    # def closest_interesection_local(self , ray : Ray) -> SurfaceRayIntersection:
    #     return self._surface.closest_intersection(ray)
    """
    
    @ti.func
    def is_inside_local(self , point : Vector) -> bool:
        return self._surface.is_inside_local(point)

class Plane(Surface):
    def __init__(self , normal , point , transfrom = Transform() , is_normal_flipped = False):
        super().__init__(transfrom , is_normal_flipped)

        self._normal = normal
        self._point = point

    @ti.func
    def closest_point_local(self ,point : Vector)->Vector:
        r = point - self._point
        return r - self._normal.dot(r) *  self._normal + self._point

    """
    # def intersects_local(self ,ray : Ray)->bool:
    #     return abs(ray.direction.dot(self._normal)) > 0

    # def bounding_box_local(self)->BoundingBox:
    #     pass
    """
    
    @ti.func
    def closest_normal_local(self , point :Vector)->Vector:
        return self._normal

    """
    # def closest_interesection_local(self , ray :Ray)->SurfaceRayIntersection:
    #     intersect = SurfaceRayIntersection()
    #     dn = ray.direction.dot(self._normal)
    #     if abs(dn) > 0:
    #         t = self._normal.dot(self._point - ray.origin) / dn
    #         if t >= 0 :
    #             intersect.is_intersecting = True
    #             intersect.distance = t
    #             intersect.point = ray.point_at(t)
    #             intersect.normal = self._normal
        
    #     return intersect
    """

class Box(Surface):
    def __init__(
        self ,
        lower_corner , 
        upper_corner , 
        transfrom = Transform() , 
        is_normal_flipped = False ):

        super().__init__(transfrom , is_normal_flipped)
        self._bound = BoundingBox(lower_corner , upper_corner)
        self._plane = [
            Plane(ti.Vector([1,0]), self._bound.upper_corner),
            Plane(ti.Vector([0,1]), self._bound.upper_corner),
            Plane(ti.Vector([-1,0]),self._bound.lower_corner),
            Plane(ti.Vector([0,-1]),self._bound.lower_corner)
        ]

    @ti.func
    def closest_point_local(self ,point : Vector)->Vector:
        result = point
        dis = 1e8
        if self._bound.contains(point):
            for i in ti.static(range(4)):
                p = self._plane[i].closest_point_local(point)
                d = distance_sqr(p , point)
                if dis > d:
                    dis = d
                    result = p
        else:
            result = clamp(point , self._bound.lower_corner , self._bound.upper_corner)

        return result

    """
    # def intersects_local(self ,ray : Ray)->bool:
    #     return self._bound.intersects(ray)

    # def bounding_box_local(self)->BoundingBox:
    #     pass
    """
    
    @ti.func
    def closest_normal_local(self , point :Vector)->Vector:
        result = self._plane[0]._normal
        if self._bound.contains(point):
            closest_n = self._plane[0]._normal
            dis = 1e8            

            for i in ti.static(range(4)):
                local_p = self._plane[i].closest_point_local(point)
                local_dis = distance_sqr(local_p , point)

                if local_dis < dis :
                    closest_n = self._plane[i]._normal
                    dis = local_dis
            result = closest_n
        else:
            point_to_cloest = point - clamp(point , self._bound.lower_corner,self._bound.upper_corner)
            closest_n = self._plane[0]._normal
            angle = closest_n.dot(point_to_cloest)

            for i in ti.static(range(1,4)):
                cos = self._plane[i]._normal.dot(point_to_cloest)
                if cos > angle :
                    closest_n = self._plane[i]._normal
                    angle = cos
            result = closest_n

        return result

    @ti.func
    def is_inside_local(self , point : Vector)->bool:
        return self._bound.contains(point)

    """
    # def closest_interesection_local(self , ray :Ray)->SurfaceRayIntersection:
    #     pass
    """

class Ball(Surface):
    def __init__(self , mid_point , radius):
        self._mid = mid_point
        self._radius = abs(radius)

    @ti.func
    def closest_point_local(self , point : Vector)->Vector:
        return self._mid + self.closest_normal_local(point) * self._radius

    @ti.func
    def is_inside_local(self , point : Vector)->bool:
        return distance_sqr(point , self._mid) < (self._radius ** 2)

    @ti.func
    def closest_normal_local(self , point :Vector )->Vector:
        return (point - self._mid).normalized()

    @ti.func
    def closest_distance_local(self ,point : Vector)->Float:
        dis = self._radius - distance(point , self._mid)
        return abs(dis)
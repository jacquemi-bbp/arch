"""
Geometry module that contains geometric functions
"""

# Copyright (C) 2021  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from math import sqrt

import numpy as np
from shapely import geometry, MultiPoint
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import split


def distance(pt1, pt2):
    """
    Return the euclidian distance from two 2D points
    Args:
        pt1:(np.array) of shape (2,): X,Y coordinates
        pt2:(np.array) of shape (2,): X,Y coordinates
    Returns:
          float : The euclidian distance  form p1 to p2

    """
    return sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_extrapoled_segement(segment_endpoints_coordinates, extrapol_ratio=1.5):
    """
    Extrapolates a segment in both directions of extrapol_ratio ratio
    Args:
        segment_endpoints_coordinates: (np.array) of shape(2, 2): coordinates of the segment's
        endpoints
        extrapol_ratio:(float) Ratio used to extrapolate the segment endpoints coordianates
    Returns:
         (np.array) of shape(2, 2) the extrapolated segment's endpoints
    """
    pt1 = segment_endpoints_coordinates[0]
    pt2 = segment_endpoints_coordinates[1]
    new_pt1 = (
        pt2[0] + extrapol_ratio * (pt1[0] - pt2[0]),
        pt2[1] + extrapol_ratio * (pt1[1] - pt2[1]),
    )
    new_pt2 = (
        pt1[0] + extrapol_ratio * (pt2[0] - pt1[0]),
        pt1[1] + extrapol_ratio * (pt2[1] - pt1[1]),
    )
    return np.array([new_pt1, new_pt2])


def create_grid(
    top_left, top_right, bottom_left, bottom_right, s1_coordinates, nb_row, nb_col
):
    """
    Create a grid on a polygon defined by s1_coordinates.
    - The vertical lines are straight. There endpoints coordinates are computed from the
         quadrilateral top and bottom
     lines in order to split them in a regular way.
    - The horizontal "lines" are composed of several segments that "follow" the quadrilateral
         top and bottom
     lines shape to represent the brain's depth

    Args:
        points_annotations:(np.array) of shape(5, 2) containing the following points coordinates(mm)
                        direction:top_left, top_right, bottom_right, bottom_left, top_left
        s1_coordinates:(np.array) shape (nb_vertices, 2) containing S1 polygon coordinates (mm)
        nb_row:(int) grid number of rows
        nb_col:(int) grid number of columns
    Returns:
         tuple:
        - list of horizontal LineString that defined the grid

    """
    vertical_lines = vertical_line_splitter(
        top_left, top_right, bottom_left, bottom_right, s1_coordinates, nb_col
    )
    return horizontal_line_splitter(vertical_lines, nb_row), vertical_lines


def vertical_line_splitter(
    top_left, top_right, bottom_left, bottom_right, s1_coordinates, nb_col
):
    """
    Create some vertical lined on a polygon defined by s1_coordinates.
    - The vertical lines are straight. There endpoints coordinates are computed from the
     quadrilateral top and bottom
     lines in order to split them in a regular way.

    Args:
        points_annotations_dataframe:(dataframe) containing the following points coordinates (mm)
                        in clockwise direction: top_left, top_right, bottom_right, bottom_left,
                        top_left
        s1_coordinates:(np.array) of shape (nb_vertices, 2) containing S1 polygon coordinates
                          (mm)
        nb_col:(int) number of columns

    Returns:
          list of vertical LineString
    """
    # Vertical lines

    vertical_lines = [
        LineString(
            [[top_left[0] - 2000, top_left[1]], [bottom_left[0] - 2000, bottom_left[1]]]
        )
    ]

    for i in range(nb_col - 1):
        top_point = top_left + (top_right - top_left) / nb_col * (i + 1)
        bottom_point = bottom_left + (bottom_right - bottom_left) / nb_col * (i + 1)
        line_coordinates = get_extrapoled_segement(
            [(top_point[0], top_point[1]), (bottom_point[0], bottom_point[1])], 1.3
        )
        intersection_line = LineString(line_coordinates).intersection(
            Polygon(s1_coordinates)
        )
        if isinstance(intersection_line, MultiLineString):
            for line in intersection_line.geoms:
                vertical_lines.append(line)
                break
        else:
            vertical_lines.append(intersection_line)
    vertical_lines.append(
        LineString(
            [
                [top_right[0] + 2000, top_right[1]],
                [bottom_right[0] + 2000, bottom_right[1]],
            ]
        )
    )
    return vertical_lines


def horizontal_line_splitter(vertical_lines, nb_row):
    """
    Create a grid on a polygon defined by s1_coordinates.
    - The vertical lines are straight. There endpoints coordinates are computed from the
          quadrilateral top and bottom
     lines in order to split them in a regular way.
    - The horizontal "lines" are composed of several segments that "follow" the quadrilateral
          top and bottom
     lines shape to represent the brain's depth

    Args:
        vertical_lines:(list) list of vertical LineString that defined the grid
        nb_row:(int) grid number of rows
    Returns:
         list of horizontal LineString
    """
    horizontal_lines = []
    for i in range(nb_row - 1):
        horizontal_points = []
        for line in vertical_lines:
            line_coords = np.array(line.coords)
            point = line_coords[0] + (line_coords[1] - line_coords[0]) / nb_row * (
                i + 1
            )
            horizontal_points.append(point)

        horizontal_line = LineString(horizontal_points)
        horizontal_lines.append(horizontal_line)
    return horizontal_lines


def create_depth_polygons(s1_coordinates, horizontal_lines):
    """
    Create shapely polygon defined by horizontal lines and the polygon defined
     by s1_coordinates
     Args:
        s1_coordinates:(np.array) of shape (nb_vertices, 2) containing
                          S1 polygon coordinates (mm)
        horizontal_lines: list of horizontal LineString that defined the grid
    Returns:
         list of shapely polygons representing S1 layers as fonction
             if brain depth
    """
    # try:
    split_polygons = []
    polygon_to_split = Polygon(s1_coordinates)
    for line in horizontal_lines:
        split_result = split(polygon_to_split, line).geoms
        try:
            polygon_to_split = split_result[1]
            split_polygons.append(split_result[0])
        except IndexError:
            pass

    split_polygons.append(polygon_to_split)
    return split_polygons


def count_nb_cell_per_polygon(cells_centroid_x, cells_centroid_y, split_polygons):
    """
    Count the number of cells located inside each polygon of split_polygons list
    Args:
        cells_centroid_x:np.array of shape (number of cells, ) of type float
        cells_centroid_y:np.array of shape (number of cells, ) of type float
        split_polygons:list of shapely polygons representing S1 layers as
                          function if brain depth
    Returns:
         list of int:The number of cells located inside each polygons of
         split_polygons
    """
    nb_cell_per_polygon = [0] * len(split_polygons)
    for x_coord, y_coord in zip(cells_centroid_x, cells_centroid_y):
        for index, polygon in enumerate(split_polygons):
            if polygon.contains(Point([x_coord, y_coord])):
                nb_cell_per_polygon[index] += 1
    return nb_cell_per_polygon


def compute_cells_polygon_level(split_polygons, cells_centroid_x, cells_centroid_y):
    """
    Compute cells polygon level
    Args:
        split_polygons:(list) list of shapely polygons representing S1 layers as
                           function if brain depth
        cells_centroid_x:(np.array) of shape (number of cells, ) of type float
        cells_centroid_y:(np.array) of shape (number of cells, ) of type float
    Returns:
        list of float that represent the polygon level for each cell

    """
    levels = [-1] * len(cells_centroid_x)
    for cell_index, (x_coord, y_coord) in enumerate(
        zip(cells_centroid_x, cells_centroid_y)
    ):
        for index, polygon in enumerate(split_polygons):
            if polygon.contains(Point([x_coord, y_coord])):
                levels[cell_index] = index
    return levels


def get_bigger_polygon(multipolygon: MultiPolygon) -> Polygon:
    """
    returns the bigger polygon within a MultiPolygon
    Args:
        multipolygon:(MultiPolygon)
    Returns:
        Polygon: the bigger polygon
    """
    polygon = None
    for poly in multipolygon.geoms:
        if polygon is None:
            polygon = poly
        else:
            if polygon.area < poly.area:
                polygon = poly
    return polygon


def get_inside_points(polygon: Polygon, points: np.array) -> np.array:
    """
    returns an array of points located inside a polygon
    Args:
        polygon:(Polygon)
        points:(np.array)
    Returns:
        a np.array of points that are located inside the polygon
    """
    inside_points = []
    for point in points:
        shapely_point = geometry.Point([point[0], point[1]])
        if polygon.contains(shapely_point):
            inside_points.append(point)
    return np.array(inside_points)


def get_layer_thickness(cell_pos):
    """
    Finds the minimum rotated rectangle inside a cloud of points which constitutes
    the layer and returns the rectangle width that represents the layer thickness
    Args:
        cell_pos:(np.array) of shape (N, 2). X/Y cells coordinates
    Returns:
        a float that correspond to the minimum_rotated_rectangle width
    """
    rectangle = MultiPoint(cell_pos).convex_hull.minimum_rotated_rectangle
    rect_coord = rectangle.boundary.coords
    linestrings = [
        LineString(rect_coord[k : k + 2]) for k in range(len(rect_coord) - 1)
    ]
    min_dist = -1
    for line in linestrings:
        if min_dist == -1 or line.length < min_dist:
            min_dist = line.length
    return min_dist

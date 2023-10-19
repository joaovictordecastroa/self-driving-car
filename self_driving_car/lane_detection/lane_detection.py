import cv2
import sys
import numpy as np
import imutils


class LaneDetection():
    def __init__(self) -> None:
        pass

    def _draw_lines(self, img, lines):
        try:
            for line in lines:
                coords = line[0]
                cv2.line(img, (coords[0], coords[1]),
                         (coords[2], coords[3]), (20, 255, 57), 2)
        except:
            pass

    def _roi(self, image, vertices):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [vertices], 255)
        masked = cv2.bitwise_and(image, mask)
        return masked

    def _remove_countour(self, img):
        image = img
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        rect_areas = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            rect_areas.append(w * h)
        avg_area = np.mean(rect_areas)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cnt_area = w * h
            if cnt_area < 0.3 * avg_area:
                image[y:y + h, x:x + w] = 0

        return image

    def _select_vertical_lines(self, lines, min_threshold=0, max_threshold=100):
        thetas = []
        best_lines = []

        for line in lines:
            ey = line[0][3]
            iy = line[0][1]
            ex = line[0][2]
            ix = line[0][0]
            dy = ey - iy
            dx = ex - ix
            theta = np.arctan(dy/dx) * 180/np.pi
            abs_theta = np.abs(theta)

            if abs_theta > min_threshold and abs_theta < max_threshold:
                best_lines.append(line)
                thetas.append(theta)

        return best_lines, thetas

    def _select_lanes(self, lines, threshold=200):
        def _get_x1(line):
            return line[0][0]

        best_lines = lines

        best_lines.sort(key=_get_x1)

        best_lines = [best_lines[0], best_lines[-1]]

        distance_between_lines = best_lines[-1][-1][0] - best_lines[0][0][0]

        return [best_lines[0]] if distance_between_lines < threshold else best_lines

    def _draw_vertices(self, img, vertices):
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 0), 1)

    def get_lanes(self, image, min_threshold, max_threshold, kernel_size):
        scale = 1

        img = cv2.GaussianBlur(image, (7, 7), 0)
        img = cv2.Canny(img, min_threshold, max_threshold)

        vertices = np.array([[0, 600], [0, 500], [266, 375], [
            532, 250], [1000, 500], [1000, 600]], np.int32)
        vertices = vertices.astype(np.int32)

        img = self._roi(img, vertices)

        thetas = []

        lines = cv2.HoughLinesP(img, 1, np.pi / 90, 180, np.array([]), 15, kernel_size)

        if lines is not None:
            lines, thetas = self._select_vertical_lines(lines)

            if len(lines) >= 2:
                lines = self._select_lanes(lines)
            if len(lines) == 1:
                lines = [lines[0], None]
            if len(lines) == 0:
                lines = None

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            self._draw_lines(img, lines)

            self._draw_vertices(img, vertices)

        return img, lines, thetas

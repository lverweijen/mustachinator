import sys
import os.path
import math

import cv2
import numpy as np
from scipy import ndimage

from filters import SimpleKalmanFilter
from shapes import Rect

# How many updates per second
FPS = 30

# Which mustache to use
MUSTACHE = os.path.join("mustaches", "mustache.png")

# Cascade files
CASCADEPATH = "cascades"
FACECASCADE = os.path.join(CASCADEPATH, "haarcascade_frontalface_default.xml")
NOSECASCADE = os.path.join(CASCADEPATH, "haarcascade_mcs_nose.xml")
MOUTHCASCADE = os.path.join(CASCADEPATH, "haarcascade_mcs_mouth.xml")
EYECASCADE = os.path.join(CASCADEPATH, "haarcascade_eye_tree_eyeglasses.xml")


class MustacheRecognizer(object):
    """Model of mustache location

    Actually we want the place to put mustache and assume there is no existing
    one."""

    def __init__(self, black=False):
        """Initialize recognizer.

        black - whether the mustache should be drawn in black. Otherwise hair
        colour is recognised."""
        self.load_resources()
        self.reset()
        self.black = black

    def reset(self):
        """Reset the state the mustache detector is in"""
        self.frame = None
        self.face = None
        self.nose = None
        self.mouth = None
        self.mustache = None
        self.eyes = [None, None]
        self.rotation = 0
        self.skin_color = None
        self.skin_std = None
        self.hair_color = None

        self.reset_filter()

        self.step = 0
        self.features = [
            self.find_face,
            self.find_nose,
            self.find_mustache,
            self.find_mouth,
            self.find_eyes
        ]

        self.nose_timeout = 0
        self.mouth_timeout = 0

    def reset_filter(self):
        """Reset kalman filter."""
        self.moustache_filter = SimpleKalmanFilter(.1)

    def load_resources(self):
        """Load cascades used for recognition and image of mustache."""
        self.mustache_image = cv2.imread(MUSTACHE, -1)

        self.faceCascade = cv2.CascadeClassifier(FACECASCADE)
        self.noseCascade = cv2.CascadeClassifier(NOSECASCADE)
        self.mouthCascade = cv2.CascadeClassifier(MOUTHCASCADE)
        self.eyeCascade = cv2.CascadeClassifier(EYECASCADE)

    def update_camera(self):
        """Read next camera frame and store it."""
        self.frame = self.cam.read()[1]
        return self.frame

    def find_face(self):
        """Perform face recognition and return if successful"""
        if self.frame is not None:
            faces = self.faceCascade.detectMultiScale(
                    self.frame, 1.2, 1,
                    minSize=(100, 100),
                    flags=cv2.cv.CV_HAAR_DO_CANNY_PRUNING)
            if len(faces) > 0:
                face = faces[0]
                self.face = Rect._make(face)
                assert(face is not None)
                return True

        return False

    def find_nose(self):
        """Perform nose recognition and return if successful."""
        if self.face is not None:
            # Specify region of interest
            left = self.face.left + self.face.width / 4
            right = self.face.right - self.face.width / 4
            top = self.face.top + self.face.height / 6
            bottom = self.face.bottom - self.face.height / 6

            # Nose should be above the mouth
            if self.mouth is not None:
                bottom = self.mouth.top

            # Nose should be below the eyes
            for eye in self.eyes:
                if eye is not None:
                    top = max(top, eye.bottom)

            # Search the nose
            face_patch = self.frame[top:bottom, left:right]
            if face_patch.size > 0:
                noses = self.noseCascade.detectMultiScale(
                        face_patch, 1.2, 1,
                        minSize=(10, 10),
                        maxSize=(self.face[2] / 4, self.face[3] / 2),
                        flags=cv2.cv.CV_HAAR_DO_CANNY_PRUNING)

                # Convert the found nose back to absolute coordinates
                if(len(noses) > 0):
                    self.nose = Rect._make(noses[0]).translate(left, top)
                    return True

        # If we didn't find anything
        if self.nose_timeout > 0:
            self.nose_timeout -= 1
        else:
            self.nose = None
            self.nose_timeout = 10

        return False

    def find_mouth(self):
        """Perform mouth recognition and return if successful."""
        if self.face is not None:

            # Specify region of interest
            left = self.face.left + self.face.width / 3
            right = self.face.right - self.face.width / 3
            top = self.face.top + self.face.height / 2
            bottom = self.face.bottom

            # Mouth should be below the nose
            if self.nose is not None:
                top = self.nose.bottom

            face_patch = self.frame[top:bottom, left:right]

            # Mouth should be wider than nose
            if self.nose is not None:
                minwidth = int(.7 * self.nose.width)
            else:
                minwidth = 30

            # Search for the mouth
            if face_patch.size > 0:
                mouths = self.mouthCascade.detectMultiScale(
                        face_patch,
                        1.2, 1,
                        minSize=(minwidth, 5))

                # Convert the result
                if len(mouths) > 0:
                    self.mouth = Rect._make(mouths[0]).translate(left, top)
                    return True

        # If we didn't find anything
        if self.mouth_timeout > 0:
            self.mouth_timeout -= 1
        else:
            self.mouth = None
            self.mouth_timeout = 10
        return False

    def find_mustache(self):
        """Try to find suitable mustache position and return if successful."""
        if self.face is not None and \
                (self.nose is not None or self.mouth is not None):

            # We can use nose, mouth or a combination to find the right spot
            # for a mustache.
            if self.mouth is not None and self.nose is not None:
                x = (self.mouth.center.x + self.nose.center.x) / 2
                y = (2 * self.nose.center.y + self.mouth.center.y) / 3
                width = max(self.nose.width, self.mouth.width)
                height = abs(self.nose.center.y - self.mouth.center.y) / 3
            else:
                chosen = self.mouth or self.nose
                x = chosen.center.x
                y = chosen.bottom if chosen == self.nose else chosen.top
                width = chosen.width
                height = 20

            left = int(x - .5 * width)
            right = int(x + .5 * width)
            top = int(y - .5 * height)
            bottom = int(y + .5 * height)

            self.mustache = Rect(left, top, right - left, bottom - top)

            # Apply filter on position relative towards face position
            if self.moustache_filter is not None:
                rel = self.mustache.translate(
                        -self.face.center.x,
                        -self.face.center.y)
                self.moustache_filter.correct(
                        np.array([rel.left, rel.right, rel.top, rel.bottom]))
                left, right, top, bottom = self.moustache_filter.predict()
                rel = Rect(left, top, right - left, bottom - top)
                self.mustache = rel.translate(
                        self.face.center.x,
                        self.face.center.y)

            return True
        else:
            # Since we might have lost track of mustache completely, we forget
            # it ever existed.
            self.mustache = None
            self.reset_filter()
            return False

    def find_eyes(self):
        """Try to detect eyes and return if successful."""
        if self.face is None:
            return False

        # Only search eyes in the upper half of the face
        bottom = self.face.top + 1./2. * self.face.height

        if self.nose:
            bottom = min(bottom, self.nose.top)

        # If no eyes at all
        if self.eyes[0] is None and self.eyes[1] is None:

            # Search them both
            face_patch = self.frame[self.face.top:bottom,
                                    self.face.left:self.face.right]
            eyes = self.eyeCascade.detectMultiScale(
                    face_patch, 1.2, 2,
                    minSize=(5, 2),
                    maxSize=(200, 200))

            # We found 2 eyes. Sort eyes by x coordinate
            if len(eyes) >= 2:
                eyes = [Rect._make(eye).translate(dx=self.face.left,
                                                  dy=self.face.top) for eye in eyes]
                if eyes[0].center.x > eyes[1].center.x:
                    eyes[0], eyes[1] = eyes[1], eyes[0]
                self.eyes = eyes
                return True

            # We found an eye, but which one?
            # Let's compare its position to the face center
            elif len(eyes) == 1:
                eye = Rect._make(eyes[0]).translate(self.face.left,
                        self.face.top)
                if eye.center.x < self.face.center.x:
                    self.eyes[0] = eye
                else:
                    self.eyes[1] = eye
                return True

            else:
                return False

        else:
            # Handle both eyes on their own
            self.find_single_eye(0)
            self.find_single_eye(1)

    def find_single_eye(self, index):
        """Find one of the eyes.

        index = 0 => left eye
        index = 1 => right eye
        ."""
        assert(0 <= index < 2)

        if self.face is None:
            return False

        # Set search region
        top = self.face.top
        bottom = self.face.top + 1./2. * self.face.height
        left = self.face.left
        right = self.face.right

        if self.nose:
            bottom = self.nose.top

        # Adapt search region for individual eyes
        if index == 1:
            left = self.face.center.x
            if self.eyes[0] is not None:
                left = max(left, self.eyes[0].right)
        else:
            right = self.face.center.x
            if self.eyes[1] is not None:
                right = min(right, self.eyes[1].left)

        face_patch = self.frame[top:bottom, left:right]
        if left < right:
            eyes = self.eyeCascade.detectMultiScale(face_patch, 1.2, 1,
                                                    minSize=(5, 2),
                                                    maxSize=(200, 200))
        else:
            return False

        if len(eyes) > 0:
            eye = eyes[0]

            self.eyes[index] = Rect._make(eye).translate(left, self.face.top)
            return True
        else:
            return False

    def get_rotation_from_eyes(self):
        """Determine rotation from face.

        Use positions of eyes to get it."""
        if self.eyes[0] is not None and self.eyes[1] is not None:
            p1 = self.eyes[0].center
            p2 = self.eyes[1].center
            rotation = -math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))
            self.rotation = rotation
            return rotation

    def get_skin_color(self):
        """Approximate persons (average) skin color."""
        if self.nose:
            nose = self.frame[
                    self.nose.top:self.nose.bottom,
                    self.nose.left:self.nose.right]

            # Determine average color
            self.skin_color = [int(nose[:, :, color].mean()) for color in [0, 1, 2]]
            self.skin_std = [int(nose[:, :, color].std()) for color in [0, 1, 2]]

    def get_hair_color(self):
        """Approximate the persons (average) hair color."""
        if self.face and self.skin_color is not None:
            hair = self.frame[
                    self.face.top - self.face.height / 50:
                    self.face.top + self.face.height / 50,
                    self.face.left:self.face.right]

            probable = np.any(hair[:, :] < np.array(self.skin_color) -
                              2 * np.array(self.skin_std), 2)
            if probable.size > 0 and hair.size > 0:
                hair = hair[probable]
            self.hair_color = [hair[:, color].mean() for color in [0, 1, 2]]
            return True
        else:
            return False

    def draw_mustache(self):
        """Draws a mustache on the found location"""
        if self.mustache is not None:
            _, _, width, height = self.mustache
            scaled_stache = cv2.resize(self.mustache_image, (int(width), int(height)))

            # Drawing the mustache is trickier than it seems
            if scaled_stache is not None:
                self.get_rotation_from_eyes()
                rotated_stache = ndimage.interpolation.rotate(
                        scaled_stache,
                        self.rotation,
                        axes=(0, 1),
                        order=0,
                        reshape=True)

                # The fourth channel is the alpha value. I consider all alpha
                # values above 0.5 to be visible.
                visible = (rotated_stache[:, :, 3] > .5).nonzero()

                # Translate the visible positions towards the position where
                # the mustache should be placed
                visible2 = tuple(visible + np.array([self.mustache.top,
                                 self.mustache.left]).reshape(2, 1))

                # There is a very small probability that it will try drawing
                # the mustache out of the window.
                try:
                    if rotated_stache is not None:
                        if self.hair_color:
                            self.output[visible2] = np.array(self.hair_color).reshape(1, 3)
                        else:
                            self.output[visible2] = rotated_stache[:, :, :3][visible]
                except Exception, e:
                    sys.stderr.write("I can't draw a mustache at that location\n")
                    sys.stderr.write("{}\n".format(e))

    def debug(self):
        """Draw bounding boxes around detected facial parts."""
        if self.face is not None:
            draw_rectangle(self.output, *self.face, color=0x0000FF)
        if self.nose is not None:
            draw_rectangle(self.output, *self.nose, color=0x00FF00)
        if self.mustache is not None:
            draw_rectangle(self.output, *map(int, self.mustache),
                           color=0xFF0000)
        if self.mouth is not None:
            draw_rectangle(self.output, *self.mouth, color=0xFFFF00)
        if self.eyes[0] is not None:
            draw_rectangle(self.output, *self.eyes[0], color=0xFFFFFF)
        if self.eyes[1] is not None:
            draw_rectangle(self.output, *self.eyes[1], color=0xFFFFFF)
        self.get_skin_color()

    def update(self, fast_update=False):
        """Run detectors.

        If fast_update is true, the program becomes faster by only running one
        of the detectors on a step."""
        self.output = self.frame.copy()

        if fast_update:
            self.features[self.step % len(self.features)]()
        else:
            for feature in self.features:
                feature()

        # Enable these lines to determine the hair color automatically
        # Otherwise a black mustache is shown
        if not self.black:
            self.get_skin_color()
            self.get_hair_color()

        self.step += 1


def draw_rectangle(frame, x, y, width, height, color=255):
    """Draw a rectangle at the given coordinates"""
    cv2.rectangle(frame, (x, y), (x + width, y + height), color)

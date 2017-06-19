import collections

# Below I defined some functions to wrap the tuples returned by the opencv
# functions.


class Point(collections.namedtuple("Point", "x y")):
    "A simple immutable Point."

    __slots__ = ()

    def translate(self, dx=0, dy=0):
        return self._replace(x=self.x + dx, y=self.y + dy)


class Rect(collections.namedtuple("Rect", "left top width height")):
    """A simple immutable Rect."""

    __slots__ = ()

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return Point(self.left + self.width / 2, self.top + self.height / 2)

    def translate(self, dx=0, dy=0):
        return self._replace(left=self.left + dx, top=self.top + dy)

    def collides(self, other):
        other = Rect._make(other)

        if self.right < other.left or self.left > other.right:
            return False
        elif self.bottom < other.top or self.top > other.bottom:
            return False
        else:
            return True

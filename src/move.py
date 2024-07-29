class Move:
    """
    contains representation for a move in game
    """
    def __init__(self, startx, starty, destx, desty):
        """
        represents moving troops from (startx, starty) to (destx, desty)

        :param startx: x coordinate of start tile
        :param starty: y coordinate of start tile
        :param destx: x coordinate of destination tile
        :param desty: y coordinate of destination tile
        """
        self.startx = startx
        self.starty = starty
        self.destx = destx
        self.desty = desty

    def __eq__(self, other):
        return (self.startx == other.startx
                and self.starty == other.starty
                and self.destx == other.destx
                and self.desty == other.desty)

    def __hash__(self):
        return hash((self.startx, self.starty, self.destx, self.desty))

    def __str__(self):
        return "start: ({}, {}), end: ({}, {})".format(self.startx, self.starty, self.destx, self.desty)
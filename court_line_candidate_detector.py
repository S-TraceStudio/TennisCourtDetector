

class CourtLineCandidateDetector:
    class Parameters:
        def __init__(self):
            self.houghThreshold = 240
            self.distanceThreshold = 8 # in pixels
            self.refinementIterations = 5

            def __init__(self, parameters=None):
                if parameters is None:
                    self.parameters = self.Parameters()
                else:
                    self.parameters = parameters

    debug = False







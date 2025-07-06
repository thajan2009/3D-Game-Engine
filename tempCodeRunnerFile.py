def renderObjects(self):
    for triangles in objects.values:
        for triangle in triangles:
            new_points = []
            for point in triangle:
                x, y, z = point

                # x diff
                ax = x / (abs(z * math.tan(math.radians(FOV/2)) - z * math.tan(math.radians(-FOV/2)))/2)
                new_points.append(ax)

                # y diff
                ay = y / (abs(z * math.tan(math.radians(FOVY/2)) - z * math.tan(math.radians(-FOVY/2)))/2)
                new_points.append(ay)

            createTriangle(self, *new_points)
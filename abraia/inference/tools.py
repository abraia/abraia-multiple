
from abraia.inference.ops import count_objects, segments_intersect, point_in_polygon


class LineCounter:
    def __init__(self, line):
        self.line = line
        self.in_count = 0
        self.out_count = 0
        self.prev_results = {}

    def update(self, results):
        for result in results:
            track_id = result.get('track_id')
            if track_id is not None:
                prev_result = self.prev_results.get(track_id)
                if prev_result:
                    (x1, y1, w1, h1), (x2, y2, w2, h2) = result['box'], prev_result['box']
                    val = segments_intersect(self.line[0], self.line[1], [x1 + w1 / 2, y1 + h1 / 2], [x2 + w2 / 2, y2 + h2 / 2])
                    if val > 0:
                        self.in_count += 1
                    if val < 0:
                        self.out_count += 1
                self.prev_results[track_id] = result
        return self.in_count, self.out_count


class RegionFilter:
    def __init__(self, polygon):
        self.region = polygon

    def update(self, results):
        in_objects, out_objects = [], []
        for result in results:
            box = result.get('box')
            if box:
                x, y, w, h = box
                center = x + w / 2, y + h / 2
                if point_in_polygon(center, self.region):
                    in_objects.append(result)
                else:
                    out_objects.append(result)
        return in_objects, out_objects


class RegionTimer():
    def __init__(self, polygon):
        self.region = polygon
        self.timers = {}

    def update(self, results, frame_time):
        in_objects, out_objects = [], []
        for result in results:
            box = result.get('box')
            track_id = result.get('track_id')
            start_time = self.timers.get(track_id)
            if box and track_id is not None:
                x, y, w, h = box
                center = x + w / 2, y + h / 2
                if point_in_polygon(center, self.region):
                    in_objects.append(result)
                    if start_time == None:
                        self.timers[track_id] = frame_time
                    else:
                        print(f'Time in zone [{track_id}]: {round(frame_time - start_time, 2)}')
                        result['label'] = f"waiting {round(frame_time - start_time, 2)}s"
                        del result['score']
                else:
                    out_objects.append(result)
                    if start_time != None:
                        del self.timers[track_id]
        return in_objects, out_objects

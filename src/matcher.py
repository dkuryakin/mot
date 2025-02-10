import json
import sys
from typing import NamedTuple


class Interval(NamedTuple):
    start: int
    end: int


class Track(NamedTuple):
    label: str
    intervals: list[Interval]

    @classmethod
    def from_json(cls, filename: str) -> list["Track"]:
        with open(filename, "r") as f:
            data = json.load(f)
        return [
            cls(
                label=label,
                intervals=[
                    Interval(start=interval[0], end=interval[1])
                    for interval in intervals
                ],
            )
            for label, intervals in data.items()
        ]

    @property
    def normalized_intervals(self) -> list[Interval]:
        sorted_intervals = sorted(self.intervals, key=lambda x: x.start)
        for i, interval in enumerate(sorted_intervals[1:]):
            if interval.start <= sorted_intervals[i].end:
                raise ValueError(f"Overlapping intervals: {interval} and {sorted_intervals[i]}")
        return sorted_intervals

    def __and__(self, track: "Track") -> float:
        return self.iou(track)

    def iou(self, track: "Track") -> float:
        a = self.normalized_intervals
        b = track.normalized_intervals

        sum_a = sum(interval.end - interval.start + 1 for interval in a)
        sum_b = sum(interval.end - interval.start + 1 for interval in b)

        if sum_a + sum_b == 0:
            return 0.0

        events = []

        for interval in a:
            start = interval.start
            end = interval.end
            events.append((start, 1, 0))
            events.append((end + 1, -1, 0))

        for interval in b:
            start = interval.start
            end = interval.end
            events.append((start, 0, 1))
            events.append((end + 1, 0, -1))

        def event_key(e):
            pos, da, db = e
            is_end = da < 0 or db < 0
            return (pos, 0 if is_end else 1)

        events.sort(key=event_key)

        prev_pos = None
        a_count = 0
        b_count = 0
        intersection = 0

        for event in events:
            current_pos, delta_a, delta_b = event
            if prev_pos is not None and current_pos > prev_pos:
                duration = current_pos - prev_pos
                if a_count > 0 and b_count > 0:
                    intersection += duration

            a_count += delta_a
            b_count += delta_b
            prev_pos = current_pos

        union = sum_a + sum_b - intersection
        if union == 0:
            return 0.0

        return intersection / union


class TrackMatch(NamedTuple):
    track_a: Track
    track_b: Track
    iou: float


class TracksMatch(NamedTuple):
    matches: list[TrackMatch]
    unmatched_a: list[Track]
    unmatched_b: list[Track]

    @property
    def score(self) -> float:
        return sum(match.iou for match in self.matches) / (
                len(self.matches) + len(self.unmatched_a) + len(self.unmatched_b))

    def __repr__(self) -> str:
        pretty = []
        for match in self.matches:
            pretty.append(f"[+] {match.track_a.label} -> {match.track_b.label} (IOU: {match.iou})")
        for track in self.unmatched_a:
            pretty.append(f"[-] {track.label} -> ?")
        for track in self.unmatched_b:
            pretty.append(f"[-] ? -> {track.label}")
        pretty.append(f"Score: {self.score}")
        return "\n".join(pretty)

    def __str__(self) -> str:
        return self.__repr__()


def match_tracks(tracks_a: list[Track], tracks_b: list[Track]) -> TracksMatch:
    a = {track.label: track for track in tracks_a}
    b = {track.label: track for track in tracks_b}

    final_matches = []
    while a and b:
        matches = []
        for label_a, track_a in a.items():
            for label_b, track_b in b.items():
                iou = track_a & track_b
                matches.append(TrackMatch(track_a, track_b, iou))

        matches.sort(key=lambda x: x.iou, reverse=True)

        best_match = matches[0]
        if best_match.iou == 0.0:
            break

        final_matches.append(best_match)

        del a[best_match.track_a.label]
        del b[best_match.track_b.label]

    return TracksMatch(
        matches=final_matches,
        unmatched_a=list(a.values()),
        unmatched_b=list(b.values()),
    )


if __name__ == "__main__":
    tracks_a = Track.from_json(sys.argv[1])
    tracks_b = Track.from_json(sys.argv[2])
    print(match_tracks(tracks_a, tracks_b))
